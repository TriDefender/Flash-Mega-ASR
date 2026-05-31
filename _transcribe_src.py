    @torch.no_grad()
    def transcribe(
        self,
        audio: Union[AudioLike, List[AudioLike]],
        context: Union[str, List[str]] = "",
        language: Optional[Union[str, List[Optional[str]]]] = None,
        return_time_stamps: bool = False,
    ) -> List[ASRTranscription]:
        """
        Transcribe audio with optional context and optional forced alignment timestamps.

        Args:
            audio:
                Audio input(s). Supported:
                  - str: local path / URL / base64 data url
                  - (np.ndarray, sr)
                  - list of above
            context:
                Context string(s). If scalar, it will be broadcast to batch size.
            language:
                Optional language(s). If provided, it must be in supported languages.
                If scalar, it will be broadcast to batch size.
                If provided, the prompt will force output to be transcription text only.
            return_time_stamps:
                If True, timestamps are produced via forced aligner and merged across chunks.
                This requires forced_aligner initialized.

        Returns:
            List[ASRTranscription]: One result per input audio.

        Raises:
            ValueError:
                - If return_time_stamps=True but forced_aligner is not provided.
                - If language is unsupported.
                - If batch sizes mismatch for context/language.
        """
        if return_time_stamps and self.forced_aligner is None:
            raise ValueError("return_time_stamps=True requires `forced_aligner` to be provided at initialization.")

        wavs = normalize_audios(audio)
        n = len(wavs)

        ctxs = context if isinstance(context, list) else [context]
        if len(ctxs) == 1 and n > 1:
            ctxs = ctxs * n
        if len(ctxs) != n:
            raise ValueError(f"Batch size mismatch: audio={n}, context={len(ctxs)}")

        langs_in: List[Optional[str]]
        if language is None:
            langs_in = [None] * n
        else:
            langs_in = language if isinstance(language, list) else [language]
            if len(langs_in) == 1 and n > 1:
                langs_in = langs_in * n
            if len(langs_in) != n:
                raise ValueError(f"Batch size mismatch: audio={n}, language={len(langs_in)}")

        langs_norm: List[Optional[str]] = []
        for l in langs_in:
            if l is None or str(l).strip() == "":
                langs_norm.append(None)
            else:
                ln = normalize_language_name(str(l))
                validate_language(ln)
                langs_norm.append(ln)

        max_chunk_sec = MAX_FORCE_ALIGN_INPUT_SECONDS if return_time_stamps else MAX_ASR_INPUT_SECONDS

        # chunk audios and record mapping
        chunks: List[AudioChunk] = []
        for i, wav in enumerate(wavs):
            parts = split_audio_into_chunks(
                wav=wav,
                sr=SAMPLE_RATE,
                max_chunk_sec=max_chunk_sec,
            )
            for j, (cwav, offset_sec) in enumerate(parts):
                chunks.append(AudioChunk(orig_index=i, chunk_index=j, wav=cwav, sr=SAMPLE_RATE, offset_sec=offset_sec))

        # run ASR on chunks
        chunk_ctx: List[str] = [ctxs[c.orig_index] for c in chunks]
        chunk_lang: List[Optional[str]] = [langs_norm[c.orig_index] for c in chunks]
        chunk_wavs: List[np.ndarray] = [c.wav for c in chunks]
        raw_outputs = self._infer_asr(chunk_ctx, chunk_wavs, chunk_lang)

        # parse outputs, prepare for optional alignment
        per_chunk_lang: List[str] = []
        per_chunk_text: List[str] = []
        for out, forced_lang in zip(raw_outputs, chunk_lang):
            lang, txt = parse_asr_output(out, user_language=forced_lang)
            per_chunk_lang.append(lang)
            per_chunk_text.append(txt)

        # forced alignment (optional)
        per_chunk_align: List[Optional[Any]] = [None] * len(chunks)
        if return_time_stamps:
            to_align_audio = []
            to_align_text = []
            to_align_lang = []
            to_align_idx = []

            for idx, (c, txt, lang_pred) in enumerate(zip(chunks, per_chunk_text, per_chunk_lang)):
                if txt.strip() == "":
                    continue
                to_align_audio.append((c.wav, c.sr))
                to_align_text.append(txt)
                to_align_lang.append(lang_pred)
                to_align_idx.append(idx)

            # batch align with max_inference_batch_size
            aligned_results: List[Any] = []
            for a_chunk, t_chunk, l_chunk in zip(
                chunk_list(to_align_audio, self.max_inference_batch_size),
                chunk_list(to_align_text, self.max_inference_batch_size),
                chunk_list(to_align_lang, self.max_inference_batch_size),
            ):
                aligned_results.extend(
                    self.forced_aligner.align(audio=a_chunk, text=t_chunk, language=l_chunk)
                )

            # offset fix
            for k, idx in enumerate(to_align_idx):
                c = chunks[idx]
                r = aligned_results[k]
                per_chunk_align[idx] = self._offset_align_result(r, c.offset_sec)

        # merge chunks back to original samples
        out_langs: List[List[str]] = [[] for _ in range(n)]
        out_texts: List[List[str]] = [[] for _ in range(n)]
        out_aligns: List[List[Any]] = [[] for _ in range(n)]

        for c, lang, txt, al in zip(chunks, per_chunk_lang, per_chunk_text, per_chunk_align):
            out_langs[c.orig_index].append(lang)
            out_texts[c.orig_index].append(txt)
            if return_time_stamps and al is not None:
                out_aligns[c.orig_index].append(al)

        results: List[ASRTranscription] = []
        for i in range(n):
            merged_text = "".join([t for t in out_texts[i] if t is not None])
            merged_language = merge_languages(out_langs[i])
            merged_align = None
            if return_time_stamps:
                merged_align = self._merge_align_results(out_aligns[i])
            results.append(ASRTranscription(language=merged_language, text=merged_text, time_stamps=merged_align))

        return results
