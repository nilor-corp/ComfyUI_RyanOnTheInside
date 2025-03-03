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
