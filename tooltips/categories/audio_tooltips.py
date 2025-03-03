from ..tooltip_manager import TooltipManager

TooltipManager.NODE_DESCRIPTIONS[
    "WaveletFeatureExtractor"
] = """
Extracts frequency band features using wavelet decomposition. Useful for analyzing different frequency components of audio.
"""

TooltipManager.NODE_TOOLTIPS["WaveletFeatureExtractor"] = {
    "audio": "Input audio to analyze",
    "wavelet_type": "Type of wavelet to use. Different wavelets capture different aspects of the signal",
    "decomposition_level": "Number of levels to decompose the signal. Higher levels give finer frequency resolution",
    "band_type": "Which frequency band to extract: high (harmonics/noise), mid (main notes), or low (bass/fundamentals)",
}

TooltipManager.NODE_DESCRIPTIONS[
    "EMDFeatureExtractor"
] = """
Extracts features using Empirical Mode Decomposition. Useful for analyzing natural oscillations in audio.
"""

TooltipManager.NODE_TOOLTIPS["EMDFeatureExtractor"] = {
    "audio": "Input audio to analyze",
    "oscillation_type": "Type of oscillation to extract: fast (tremolo/vibrato), medium (note changes), slow (phrases)",
    "num_imfs": "Number of Intrinsic Mode Functions to extract. More IMFs give finer decomposition",
}

TooltipManager.NODE_DESCRIPTIONS[
    "MelodicRangeFeatureExtractor"
] = """
Extracts melodic content features from audio using mel spectrograms. Similar to Sonic Visualiser's Melodic Range Spectrogram view.
"""

TooltipManager.NODE_TOOLTIPS["MelodicRangeFeatureExtractor"] = {
    "audio": "Input audio to analyze",
    "extraction_method": "Frequency range to focus on: low (20-500Hz), mid (200-2000Hz), high (1000-8000Hz), or full range",
    "n_mels": "Number of mel bands for the spectrogram. Higher values give more detailed frequency resolution",
}

# Add tooltips for NoteEventsFeatureExtractor
TooltipManager.NODE_TOOLTIPS["NoteEventsFeatureExtractor"] = {
    "audio": "Input audio to analyze for note events.",
    "extraction_method": "Type of note feature to extract:\n- note_onsets: Shows when notes begin\n- note_pitches: Shows the pitches of active notes\n- note_durations: Shows how long notes are held\n- note_density: Shows how many notes are active at once\n- note_activity: Shows note on/off envelope with attack/decay",
    "onset_threshold": "Sensitivity for note onset detection. Higher values make the detector less sensitive (fewer false positives but might miss quiet notes).",
    "min_note_duration": "Minimum duration in seconds for a note to be considered valid. Helps filter out noise and short artifacts.",
    "pitch_min": "Minimum frequency in Hz to consider when detecting notes. Notes below this frequency will be ignored.",
    "pitch_max": "Maximum frequency in Hz to consider when detecting notes. Notes above this frequency will be ignored.",
    "polyphony_enabled": "When enabled, detects multiple simultaneous notes. When disabled, only the most prominent note at any time is tracked.",
    "opt_crepe_model": "CREPE model to use for pitch detection (requires crepe and tensorflow). 'none' falls back to librosa. Larger models are more accurate but slower.",
    "notes_json": "JSON string containing all detected notes with their onset times, durations, pitches, and other properties. Can be used with text nodes to extract specific note data.",
}
