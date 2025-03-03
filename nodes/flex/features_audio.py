from .features import BaseFeature
import librosa
import numpy as np
from scipy.signal import hilbert
from scipy import signal
import functools
from abc import ABC, abstractmethod
import numpy as np
import librosa
from scipy.signal import medfilt, hilbert
from scipy import signal
import functools


class BaseAudioFeature(BaseFeature):

    @classmethod
    @abstractmethod
    def get_extraction_methods(cls):
        """Return a list of parameter names that can be modulated."""
        return []

    def __init__(self, name, audio, frame_count, frame_rate, width, height):
        super().__init__(name, "audio", frame_rate, frame_count, width, height)
        self.audio = audio
        self.sample_rate = None
        self.frame_duration = None
        self.feature_name = None
        self.available_features = []

    def _prepare_audio(self):
        self.sample_rate = self.audio["sample_rate"]
        waveform = self.audio["waveform"]

        # Handle multi-dimensional tensors
        if waveform.ndim > 1:
            waveform = waveform.squeeze()

        # Handle multi-channel audio by averaging channels
        if waveform.ndim > 1:
            waveform = waveform.mean(axis=0)

        self.audio_array = waveform.cpu().numpy()
        self.frame_duration = 1 / self.frame_rate

    def _get_audio_frame(self, frame_index):
        start_time = frame_index * self.frame_duration
        end_time = start_time + self.frame_duration
        start_sample = int(start_time * self.sample_rate)
        end_sample = int(end_time * self.sample_rate)
        if start_sample >= len(self.audio_array):
            return np.array([])  # Return empty array if we've run out of audio
        return self.audio_array[start_sample : min(end_sample, len(self.audio_array))]

    def get_feature_sequence(self, feature_name=None):
        if self.features is None:
            self.extract()
        if feature_name is None:
            feature_name = self.feature_name
        return self.features.get(feature_name, None)

    def set_active_feature(self, feature_name):
        if feature_name in self.available_features:
            self.feature_name = feature_name
        else:
            raise ValueError(
                f"Invalid feature name. Available features are: {', '.join(self.available_features)}"
            )


class AudioFeature(BaseAudioFeature):

    def __init__(
        self,
        feature_name,
        audio,
        frame_count,
        frame_rate,
        width,
        height,
        feature_type="amplitude_envelope",
    ):
        super().__init__(feature_name, audio, frame_count, frame_rate, width, height)
        self.feature_type = feature_type
        self.available_features = self.get_extraction_methods()
        self.feature_name = feature_type
        self._prepare_audio()

    @classmethod
    def get_extraction_methods(cls):
        """Return a list of parameter names that can be modulated."""
        return [
            "amplitude_envelope",
            "rms_energy",
            "spectral_centroid",
            "onset_strength",
            "chroma_features",
            "wavelet_high_frequency",  # High harmonics, bow noise
            "wavelet_mid_frequency",  # Main note frequencies
            "wavelet_low_frequency",  # Fundamental frequencies
            "emd_fast",  # Rapid playing techniques (tremolo)
            "emd_medium",  # Note transitions
            "emd_slow",  # Overall phrase dynamics
        ]

    def extract(self):
        self.features = {self.feature_name: []}
        for i in range(self.frame_count):
            frame = self._get_audio_frame(i)
            value = self._calculate_feature(frame)
            self.features[self.feature_name].append(value)
        self._normalize_features()
        return self

    def _calculate_feature(self, frame):
        if frame.size == 0:
            return 0.0
        if self.feature_name == "amplitude_envelope":
            return np.max(np.abs(frame))
        elif self.feature_name == "rms_energy":
            return np.sqrt(np.mean(frame**2))
        elif self.feature_name == "spectral_centroid":
            centroid = librosa.feature.spectral_centroid(y=frame, sr=self.sample_rate)
            return np.mean(centroid)
        elif self.feature_name == "onset_strength":
            strength = librosa.onset.onset_strength(y=frame, sr=self.sample_rate)
            return np.mean(strength)
        elif self.feature_name == "chroma_features":
            chroma = librosa.feature.chroma_stft(y=frame, sr=self.sample_rate)
            return np.mean(chroma)
        else:
            raise ValueError(f"Unsupported feature type: {self.feature_name}")

    def _normalize_features(self):
        feature_array = np.array(self.features[self.feature_name], dtype=np.float32)
        min_val = np.min(feature_array)
        max_val = np.max(feature_array)
        if max_val > min_val:
            normalized = (feature_array - min_val) / (max_val - min_val)
        else:
            normalized = np.zeros_like(feature_array)
        self.features[self.feature_name] = normalized.tolist()

    def _extract_wavelet_features(self):
        from ..audio.audio_utils import perform_wavelet_transform, calculate_band_energy

        waveform = self.audio["waveform"]
        sample_rate = self.audio["sample_rate"]

        # Get wavelet coefficients
        coeffs = perform_wavelet_transform(
            waveform, self.wavelet_type, self.decomposition_level
        )

        # Extract appropriate band based on feature type
        if self.feature_type == "wavelet_high_frequency":
            self.values = calculate_band_energy(coeffs[0], sample_rate, self.frame_rate)
        elif self.feature_type == "wavelet_mid_frequency":
            self.values = calculate_band_energy(coeffs[1], sample_rate, self.frame_rate)
        elif self.feature_type == "wavelet_low_frequency":
            self.values = calculate_band_energy(coeffs[2], sample_rate, self.frame_rate)

    def _extract_emd_features(self):
        from ..audio.audio_utils import perform_emd, calculate_imf_envelope

        waveform = self.audio["waveform"]
        sample_rate = self.audio["sample_rate"]

        # Get IMFs
        imfs = perform_emd(waveform)

        # Extract appropriate IMF based on feature type
        if self.feature_type == "emd_fast":
            self.values = calculate_imf_envelope(imfs[0], sample_rate, self.frame_rate)
        elif self.feature_type == "emd_medium":
            self.values = calculate_imf_envelope(imfs[1], sample_rate, self.frame_rate)
        elif self.feature_type == "emd_slow":
            self.values = calculate_imf_envelope(imfs[2], sample_rate, self.frame_rate)


import numpy as np
import librosa


class RhythmFeature(BaseAudioFeature):
    def __init__(
        self,
        feature_name,
        audio,
        frame_count,
        frame_rate,
        width,
        height,
        feature_type="beat_locations",
        time_signature=4,
    ):
        super().__init__(feature_name, audio, frame_count, frame_rate, width, height)
        self.feature_type = feature_type
        self.available_features = self.get_extraction_methods()
        self.feature_name = feature_type
        self.time_signature = time_signature  # Default to 4/4 time
        self._prepare_audio()

    @classmethod
    def get_extraction_methods(cls):
        return [
            "beat_locations",
            "tempo",
            "onset_strength",
            "beat_emphasis",
            "syncopation",
            "rhythm_regularity",
            "down_beats",
            "up_beats",
        ]

    def extract(self):
        self.features = {}

        # Extract basic rhythm features
        tempo, beat_frames = librosa.beat.beat_track(
            y=self.audio_array, sr=self.sample_rate
        )
        onset_env = librosa.onset.onset_strength(
            y=self.audio_array, sr=self.sample_rate
        )

        # Calculate features based on the selected feature_type
        if self.feature_type == "beat_locations":
            self._extract_beat_locations(beat_frames)
        elif self.feature_type == "tempo":
            self._extract_tempo(tempo)
        elif self.feature_type == "onset_strength":
            self._extract_onset_strength(onset_env)
        elif self.feature_type == "beat_emphasis":
            self._extract_beat_emphasis(beat_frames, onset_env)
        elif self.feature_type == "syncopation":
            self._extract_syncopation(beat_frames, onset_env)
        elif self.feature_type == "rhythm_regularity":
            self._extract_rhythm_regularity(beat_frames)
        elif self.feature_type in ["down_beats", "up_beats"]:
            self._extract_beat_types(beat_frames)
        else:
            raise ValueError(f"Unsupported feature type: {self.feature_type}")

        self._normalize_features()
        return self

    def _extract_beat_locations(self, beat_frames):
        beat_times = librosa.frames_to_time(beat_frames, sr=self.sample_rate)
        beat_sequence = np.zeros(self.frame_count)
        for beat_time in beat_times:
            frame_index = int(beat_time * self.frame_rate)
            if frame_index < self.frame_count:
                beat_sequence[frame_index] = 1
        self.features[self.feature_name] = beat_sequence.tolist()

    def _extract_tempo(self, tempo):
        # For simplicity, we'll use a constant tempo for all frames
        self.features[self.feature_name] = [tempo] * self.frame_count

    def _extract_onset_strength(self, onset_env):
        # Resample onset strength to match frame count
        resampled_onset = np.interp(
            np.linspace(0, len(onset_env), self.frame_count),
            np.arange(len(onset_env)),
            onset_env,
        )
        self.features[self.feature_name] = resampled_onset.tolist()

    def _extract_beat_emphasis(self, beat_frames, onset_env):
        beat_emphasis = np.zeros(self.frame_count)
        for beat in beat_frames:
            if beat < len(onset_env):
                emphasis = onset_env[beat]
                frame_index = int(
                    librosa.frames_to_time(beat, sr=self.sample_rate) * self.frame_rate
                )
                if frame_index < self.frame_count:
                    beat_emphasis[frame_index] = emphasis
        self.features[self.feature_name] = beat_emphasis.tolist()

    def _extract_syncopation(self, beat_frames, onset_env):
        # A simple syncopation measure: difference between actual onsets and expected beat locations
        expected_beats = np.zeros_like(onset_env)
        expected_beats[beat_frames] = 1
        syncopation = np.abs(onset_env - expected_beats)

        # Resample to match frame count
        resampled_syncopation = np.interp(
            np.linspace(0, len(syncopation), self.frame_count),
            np.arange(len(syncopation)),
            syncopation,
        )
        self.features[self.feature_name] = resampled_syncopation.tolist()

    def _extract_rhythm_regularity(self, beat_frames):
        # Measure regularity by calculating the standard deviation of inter-beat intervals
        ibi = np.diff(librosa.frames_to_time(beat_frames, sr=self.sample_rate))
        regularity = 1 / (
            1 + np.std(ibi)
        )  # Invert so that higher values mean more regular
        self.features[self.feature_name] = [regularity] * self.frame_count

    def _extract_beat_types(self, beat_frames):
        beat_times = librosa.frames_to_time(beat_frames, sr=self.sample_rate)
        down_beats = np.zeros(self.frame_count)
        up_beats = np.zeros(self.frame_count)

        for i, beat_time in enumerate(beat_times):
            frame_index = int(beat_time * self.frame_rate)
            if frame_index < self.frame_count:
                if i % self.time_signature == 0:
                    down_beats[frame_index] = 1  # Down beat
                else:
                    up_beats[frame_index] = 1  # Up beat

        self.features["down_beats"] = down_beats.tolist()
        self.features["up_beats"] = up_beats.tolist()

    def _normalize_features(self):
        if self.feature_type in ["down_beats", "up_beats"]:
            # For beat types, we want to keep the original binary values
            self.features[self.feature_type] = self.features[self.feature_type]
        else:
            feature_array = np.array(self.features[self.feature_name], dtype=np.float32)
            min_val = np.min(feature_array)
            max_val = np.max(feature_array)
            if max_val > min_val:
                normalized = (feature_array - min_val) / (max_val - min_val)
            else:
                normalized = np.zeros_like(feature_array)
            self.features[self.feature_name] = normalized.tolist()

    def get_rhythm_feature(self, frame_index):
        if self.features is None:
            self.extract()

        if self.feature_type in ["down_beats", "up_beats"]:
            value = self.features[self.feature_type][frame_index]
            beat_type = self.feature_type.rstrip(
                "s"
            )  # Remove 's' to get "down_beat" or "up_beat"
            return {"value": value, "beat_type": beat_type if value > 0 else "no_beat"}
        else:
            return {"normalized": self.features[self.feature_name][frame_index]}


class PitchFeature(BaseAudioFeature):
    def __init__(
        self,
        feature_name,
        audio,
        frame_count,
        frame_rate,
        width,
        height,
        pitch_range_collections=None,
        feature_type="frequency",
        window_size=0,
        vibrato_options=None,
        crepe_model="none",
    ):
        super().__init__(feature_name, audio, frame_count, frame_rate, width, height)
        self.available_features = self.get_extraction_methods()
        self.feature_type = feature_type
        self.pitch_range_collections = pitch_range_collections or []
        self.vibrato_options = vibrato_options
        self.window_size = window_size
        self.current_frame = 0
        self.crepe_model = crepe_model
        self.feature_name = feature_name
        self._prepare_audio()

        # Parameters for pitch detection
        self.frame_length = 2048
        self.hop_length = 512
        self.fmin = 20  # Lower frequency limit
        self.fmax = 4000  # Upper frequency limit

    @staticmethod
    def calculate_tolerance(frequency, tolerance_percent):
        semitone_above = frequency * 2 ** (1 / 12)
        frequency_difference = semitone_above - frequency
        tolerance = (frequency_difference / 2) * (tolerance_percent / 100.0)
        return tolerance

    @classmethod
    def get_extraction_methods(cls):
        """Return a list of parameter names that can be modulated."""
        return [
            "frequency",
            "semitone",
            "pitch_direction",
            "vibrato_signal",
            "vibrato_strength",
        ]

    @classmethod
    def pitch_to_note(cls, pitch):
        if pitch == 0:
            return "N/A"
        return librosa.hz_to_note(pitch)

    def _calculate_pitch_sequence(self):
        # Try to use CREPE model if available
        if self.crepe_model != "none":
            try:
                import crepe
                import tensorflow

                time_steps, frequencies, confidence, activation = crepe.predict(
                    self.audio_array,
                    self.sample_rate,
                    model_capacity=self.crepe_model,
                    viterbi=True,
                    step_size=int(
                        1000 * self.hop_length / self.sample_rate
                    ),  # Step size in ms
                )
                # time_steps, frequencies, confidences, _ = crepe.predict(
                #     self.audio_array,
                #     self.sample_rate,
                #     model_capacity=self.crepe_model,
                #     viterbi=True,
                #     step_size=int(1000 * self.hop_length / self.sample_rate)  # Step size in ms
                # )
                pitches = frequencies
                confidences = confidence
            except ImportError as e:
                print(
                    f"Please 'pip install crepe tensorflow' to use any CREPE model :)"
                )
                pitches, confidences = self._fallback_pitch_estimation()
            except Exception as e:
                print(
                    f"Error using CREPE model: {e}. Falling back to librosa's pitch estimation."
                )
                pitches, confidences = self._fallback_pitch_estimation()
        else:
            # Fallback to librosa if CREPE model is not provided
            pitches, confidences = self._fallback_pitch_estimation()

        # Interpolate to match frame count
        frame_times = np.linspace(
            0, len(self.audio_array) / self.sample_rate, num=self.frame_count
        )
        if len(pitches) == 0:
            pitches = np.zeros(len(frame_times))
            confidences = np.zeros(len(frame_times))
            pitch_times = frame_times
        else:
            pitch_times = librosa.frames_to_time(
                np.arange(len(pitches)), sr=self.sample_rate, hop_length=self.hop_length
            )
            pitches = np.nan_to_num(pitches)
            confidences = np.nan_to_num(confidences)
            pitches = np.interp(frame_times, pitch_times, pitches)
            confidences = np.interp(frame_times, pitch_times, confidences)

        # Handle potential NaNs
        pitches = np.nan_to_num(pitches)
        confidences = np.nan_to_num(confidences)

        self.features = {}
        self.features[self.feature_name + "_pitch"] = pitches.tolist()
        self.features[self.feature_name + "_confidence"] = confidences.tolist()

    def _fallback_pitch_estimation(self):
        pitches, magnitudes = librosa.piptrack(
            y=self.audio_array,
            sr=self.sample_rate,
            fmin=self.fmin,
            fmax=self.fmax,
            hop_length=self.hop_length,
            threshold=0.1,
        )
        pitches = pitches.max(axis=0)
        confidences = magnitudes.max(axis=0)
        return pitches, confidences

    def extract(self):
        self.features = {}
        self._calculate_pitch_sequence()

        if self.feature_type == "frequency":
            self._extract_frequency()
        elif self.feature_type == "semitone":
            self._extract_semitone()
        elif self.feature_type == "pitch_direction":
            self._extract_pitch_direction()
        elif self.feature_type == "vibrato_signal":
            self._extract_vibrato_signal()
        elif self.feature_type == "vibrato_strength":
            self._extract_vibrato_strength()
        elif self.feature_type.startswith("wavelet_"):
            self._extract_wavelet_features()
        elif self.feature_type.startswith("emd_"):
            self._extract_emd_features()
        else:
            raise ValueError(f"Unsupported feature type: {self.feature_type}")

        self._normalize_features()
        return self

    def _extract_frequency(self):
        pitches = np.array(self.features[self.feature_name + "_pitch"])
        valid_pitches = pitches[pitches > 0]

        if len(valid_pitches) == 0:
            feature_values = np.zeros_like(pitches)
        else:
            feature_values = pitches.copy()
        self.features[self.feature_name + "_original"] = feature_values.tolist()
        self.features[self.feature_name] = feature_values.tolist()

    def _extract_semitone(self):
        pitches = np.array(self.features[self.feature_name + "_pitch"])
        midi_notes = librosa.hz_to_midi(pitches)
        midi_notes = np.round(midi_notes)
        semitone_freqs = librosa.midi_to_hz(midi_notes)

        semitone_freqs[pitches == 0] = 0.0
        self.features[self.feature_name + "_original"] = semitone_freqs.tolist()
        self.features[self.feature_name] = semitone_freqs.tolist()

    def _extract_pitch_direction(self):
        pitches = np.array(self.features[self.feature_name + "_pitch"])
        pitches[pitches == 0] = np.nan  # Replace zeros with NaN for diff calculation
        pitch_diff = np.diff(pitches, prepend=pitches[0])

        direction = np.sign(pitch_diff)
        direction = np.nan_to_num(direction)  # Replace NaNs back to zero

        self.features[self.feature_name + "_original"] = direction.tolist()
        self.features[self.feature_name] = direction.tolist()

    def _extract_vibrato_signal(self):
        pitches = np.array(self.features[self.feature_name + "_pitch"])
        pitches[pitches == 0] = np.nan  # Ignore zero pitches

        # Interpolate to fill NaNs for Hilbert transform
        valid_indices = np.where(~np.isnan(pitches))[0]
        if len(valid_indices) == 0:
            vibrato_signal = np.zeros_like(pitches)
        else:
            pitches = np.interp(
                np.arange(len(pitches)), valid_indices, pitches[valid_indices]
            )

            analytic_signal = hilbert(pitches)
            instantaneous_phase = np.unwrap(np.angle(analytic_signal))
            instantaneous_frequency = np.diff(instantaneous_phase)

            vibrato_signal = np.concatenate(([0], instantaneous_frequency))

        self.features[self.feature_name + "_original"] = vibrato_signal.tolist()
        self.features[self.feature_name] = vibrato_signal.tolist()

    def _extract_vibrato_strength(self):
        pitches = np.array(self.features[self.feature_name + "_pitch"])
        pitches[pitches == 0] = np.nan  # Ignore zero pitches

        window_size_frames = int(
            0.1 * self.sample_rate / self.hop_length
        )  # 100ms window
        if window_size_frames < 1:
            window_size_frames = 1

        # Compute standard deviation over the window
        vibrato_strength = []
        for i in range(len(pitches)):
            start = max(0, i - window_size_frames // 2)
            end = min(len(pitches), i + window_size_frames // 2)
            window_pitches = pitches[start:end]
            window_pitches = window_pitches[~np.isnan(window_pitches)]
            if len(window_pitches) > 0:
                std = np.std(window_pitches)
            else:
                std = 0.0
            vibrato_strength.append(std)

        vibrato_strength = np.array(vibrato_strength)
        self.features[self.feature_name + "_original"] = vibrato_strength.tolist()
        self.features[self.feature_name] = vibrato_strength.tolist()

    def _extract_wavelet_features(self):
        from ..audio.audio_utils import perform_wavelet_transform, calculate_band_energy
        import numpy as np
        import torch
        import sys

        waveform = self.audio["waveform"]
        sample_rate = self.audio["sample_rate"]

        # Process audio in frames to get more detailed feature variation
        feature_values = []
        hop_length = int(sample_rate / self.frame_rate)

        # Use smaller band indices - may not have enough decomposition levels
        if self.feature_type == "wavelet_high_frequency":
            band_idx = 1  # First detail for high frequency
        elif self.feature_type == "wavelet_mid_frequency":
            band_idx = 2  # Second detail for mid frequency if available
        else:  # wavelet_low_frequency
            band_idx = 0  # Approximation for low frequency

        # Process audio in frames
        for i in range(self.frame_count):
            # Get frame of audio
            frame_start = min(int(i * hop_length), waveform.shape[-1] - 1)
            frame_end = min(frame_start + hop_length, waveform.shape[-1])

            if frame_start >= frame_end:
                # Use last valid frame if we're at the end
                if i > 0 and feature_values:
                    feature_values.append(feature_values[-1])
                else:
                    feature_values.append(0.0)  # Default
                continue

            # Extract frame - CRITICAL FIX: need to extract just this segment
            if waveform.dim() == 3:  # [batch, channels, samples]
                frame = waveform[
                    0, :, frame_start:frame_end
                ]  # First batch, all channels
            elif waveform.dim() == 2:  # [channels, samples]
                frame = waveform[:, frame_start:frame_end]  # All channels
            else:  # [samples]
                frame = waveform[frame_start:frame_end]  # Just samples

            # Convert to mono if multi-channel
            if frame.dim() > 1:
                frame = torch.mean(frame, dim=0)

            # Skip empty frames
            if frame.numel() == 0:
                if i > 0 and feature_values:
                    feature_values.append(feature_values[-1])
                else:
                    feature_values.append(0.0)
                continue

            # Ensure frame is proper shape for wavelet
            if frame.dim() == 0:  # It's a scalar
                if i > 0 and feature_values:
                    feature_values.append(feature_values[-1])
                else:
                    feature_values.append(0.0)
                continue

            print(f"Frame {i} shape: {frame.shape}, numel: {frame.numel()}")

            try:
                # Perform wavelet decomposition on this specific frame
                coeffs = perform_wavelet_transform(frame, "db4", 3)  # Use 3 levels

                print(f"Frame {i} gave coeffs lengths: {[c.numel() for c in coeffs]}")

                # Check available bands and adjust if needed
                if len(coeffs) <= band_idx:
                    # Not enough decomposition levels, use highest available
                    band_idx = len(coeffs) - 1

                # Calculate energy from the wavelet coefficients
                if coeffs[band_idx].numel() > 0:
                    # Simple energy calculation - no normalization here
                    energy = torch.mean(torch.abs(coeffs[band_idx])).item()
                    # Scale to a visible range
                    energy = energy * 5.0

                    # Add some variation based on frame index to avoid flat lines in testing
                    # REMOVE THIS LINE IN PRODUCTION - it's just for debugging
                    # energy = energy * (1.0 + 0.1 * np.sin(i * 0.5))

                    feature_values.append(float(energy))
                    print(f"Frame {i} energy: {energy}")
                else:
                    # Empty coefficients
                    if i > 0 and feature_values:
                        feature_values.append(feature_values[-1])
                    else:
                        feature_values.append(0.0)
            except Exception as e:
                # Log error and use fallback value
                print(f"Error in wavelet transform for frame {i}: {e}")
                if i > 0 and feature_values:
                    feature_values.append(feature_values[-1])
                else:
                    feature_values.append(0.0)

        # Check if we have variation in our values
        if len(feature_values) > 1:
            min_val = min(feature_values)
            max_val = max(feature_values)
            print(f"Feature value range: {min_val} to {max_val}")

            if abs(max_val - min_val) < 1e-6:
                print("WARNING: No variation in feature values!")
            else:
                print(f"Variation detected: {max_val - min_val}")

        # Just store the raw values - skip normalization
        self.features = {self.feature_name: feature_values}

        return self

    def _extract_emd_features(self):
        from ..audio.audio_utils import perform_emd, calculate_imf_envelope

        waveform = self.audio["waveform"]
        sample_rate = self.audio["sample_rate"]

        # Get IMFs
        imfs = perform_emd(waveform)

        # Extract appropriate IMF based on feature type
        if self.feature_type == "emd_fast":
            self.values = calculate_imf_envelope(imfs[0], sample_rate, self.frame_rate)
        elif self.feature_type == "emd_medium":
            self.values = calculate_imf_envelope(imfs[1], sample_rate, self.frame_rate)
        elif self.feature_type == "emd_slow":
            self.values = calculate_imf_envelope(imfs[2], sample_rate, self.frame_rate)

    def _normalize_features(self):
        feature_array = np.array(self.features[self.feature_name], dtype=np.float32)
        finite_mask = np.isfinite(feature_array)
        if not np.any(finite_mask):
            normalized = np.zeros_like(feature_array)
        else:
            min_val = np.min(feature_array[finite_mask])
            max_val = np.max(feature_array[finite_mask])
            if max_val > min_val:
                normalized = np.zeros_like(feature_array)
                normalized[finite_mask] = (feature_array[finite_mask] - min_val) / (
                    max_val - min_val
                )
            else:
                normalized = np.zeros_like(feature_array)
        self.features[self.feature_name] = normalized.tolist()

    def get_pitch_feature(self, frame_index):
        if self.features is None:
            self.extract()

        original_value = self.features[self.feature_name + "_original"][frame_index]
        normalized_value = self.features[self.feature_name][frame_index]
        actual_pitch = self.features[self.feature_name + "_pitch"][frame_index]
        # Provide smoothed pitch if available
        smoothed_pitch = self.features.get(
            self.feature_name + "_smoothed", [0.0] * self.frame_count
        )[frame_index]

        return {
            "original": original_value,
            "normalized": normalized_value,
            "actual_pitch": actual_pitch,
            "smoothed_pitch": smoothed_pitch,
        }


class PitchRange:
    def __init__(self, min_pitch, max_pitch):
        self.min_pitch = min_pitch
        self.max_pitch = max_pitch

    def contains(self, pitch):
        return self.min_pitch <= pitch <= self.max_pitch


class WaveletFeature(BaseAudioFeature):
    def __init__(
        self,
        width,
        height,
        feature_name,
        audio,
        frame_count,
        frame_rate,
        feature_type="wavelet_high_frequency",
    ):
        super().__init__(feature_name, audio, frame_count, frame_rate, width, height)
        self.feature_type = feature_type
        self.available_features = self.get_extraction_methods()
        self.features = {}
        self.normalized_features = {}
        self.wavelet_type = "db4"
        self.decomposition_level = 4
        self._prepare_audio()

    @classmethod
    def get_extraction_methods(cls):
        return [
            "wavelet_high_frequency",
            "wavelet_mid_frequency",
            "wavelet_low_frequency",
        ]

    def extract(self):
        from ..audio.audio_utils import perform_wavelet_transform, calculate_band_energy
        import numpy as np
        import torch
        import sys

        waveform = self.audio["waveform"]
        sample_rate = self.audio["sample_rate"]

        # Process audio in frames to get more detailed feature variation
        feature_values = []
        hop_length = int(sample_rate / self.frame_rate)

        # Use smaller band indices - may not have enough decomposition levels
        if self.feature_type == "wavelet_high_frequency":
            band_idx = 1  # First detail for high frequency
        elif self.feature_type == "wavelet_mid_frequency":
            band_idx = 2  # Second detail for mid frequency if available
        else:  # wavelet_low_frequency
            band_idx = 0  # Approximation for low frequency

        # Process audio in frames
        for i in range(self.frame_count):
            # Get frame of audio
            frame_start = min(int(i * hop_length), waveform.shape[-1] - 1)
            frame_end = min(frame_start + hop_length, waveform.shape[-1])

            if frame_start >= frame_end:
                # Use last valid frame if we're at the end
                if i > 0 and feature_values:
                    feature_values.append(feature_values[-1])
                else:
                    feature_values.append(0.0)  # Default
                continue

            # Extract frame - CRITICAL FIX: need to extract just this segment
            if waveform.dim() == 3:  # [batch, channels, samples]
                frame = waveform[
                    0, :, frame_start:frame_end
                ]  # First batch, all channels
            elif waveform.dim() == 2:  # [channels, samples]
                frame = waveform[:, frame_start:frame_end]  # All channels
            else:  # [samples]
                frame = waveform[frame_start:frame_end]  # Just samples

            # Convert to mono if multi-channel
            if frame.dim() > 1:
                frame = torch.mean(frame, dim=0)

            # Skip empty frames
            if frame.numel() == 0:
                if i > 0 and feature_values:
                    feature_values.append(feature_values[-1])
                else:
                    feature_values.append(0.0)
                continue

            # Ensure frame is proper shape for wavelet
            if frame.dim() == 0:  # It's a scalar
                if i > 0 and feature_values:
                    feature_values.append(feature_values[-1])
                else:
                    feature_values.append(0.0)
                continue

            print(f"Frame {i} shape: {frame.shape}, numel: {frame.numel()}")

            try:
                # Perform wavelet decomposition on this specific frame
                coeffs = perform_wavelet_transform(frame, "db4", 3)  # Use 3 levels

                print(f"Frame {i} gave coeffs lengths: {[c.numel() for c in coeffs]}")

                # Check available bands and adjust if needed
                if len(coeffs) <= band_idx:
                    # Not enough decomposition levels, use highest available
                    band_idx = len(coeffs) - 1

                # Calculate energy from the wavelet coefficients
                if coeffs[band_idx].numel() > 0:
                    # Simple energy calculation - no normalization here
                    energy = torch.mean(torch.abs(coeffs[band_idx])).item()
                    # Scale to a visible range
                    energy = energy * 5.0

                    # Add some variation based on frame index to avoid flat lines in testing
                    # REMOVE THIS LINE IN PRODUCTION - it's just for debugging
                    # energy = energy * (1.0 + 0.1 * np.sin(i * 0.5))

                    feature_values.append(float(energy))
                    print(f"Frame {i} energy: {energy}")
                else:
                    # Empty coefficients
                    if i > 0 and feature_values:
                        feature_values.append(feature_values[-1])
                    else:
                        feature_values.append(0.0)
            except Exception as e:
                # Log error and use fallback value
                print(f"Error in wavelet transform for frame {i}: {e}")
                if i > 0 and feature_values:
                    feature_values.append(feature_values[-1])
                else:
                    feature_values.append(0.0)

        # Check if we have variation in our values
        if len(feature_values) > 1:
            min_val = min(feature_values)
            max_val = max(feature_values)
            print(f"Feature value range: {min_val} to {max_val}")

            if abs(max_val - min_val) < 1e-6:
                print("WARNING: No variation in feature values!")
            else:
                print(f"Variation detected: {max_val - min_val}")

        # Just store the raw values - skip normalization
        self.features = {self.feature_name: feature_values}

        return self

    def _normalize_features(self):
        for feature_name, values in self.features.items():
            min_val = min(values)
            max_val = max(values)
            range_val = max_val - min_val

            if range_val == 0:
                # Avoid division by zero
                normalized = [0.5 for _ in values]
            else:
                normalized = [(v - min_val) / range_val for v in values]

            self.normalized_features[feature_name] = normalized


class EMDFeature(BaseAudioFeature):
    def __init__(
        self,
        width,
        height,
        feature_name,
        audio,
        frame_count,
        frame_rate,
        feature_type="emd_fast",
    ):
        super().__init__(feature_name, audio, frame_count, frame_rate, width, height)
        self.feature_type = feature_type
        self.available_features = self.get_extraction_methods()
        self.features = {}
        self.normalized_features = {}
        self._prepare_audio()

    @classmethod
    def get_extraction_methods(cls):
        return [
            "emd_fast",
            "emd_medium",
            "emd_slow",
        ]

    def extract(self):
        import numpy as np
        import torch
        from scipy.signal import butter, filtfilt, hilbert

        waveform = self.audio["waveform"]
        sample_rate = self.audio["sample_rate"]

        # Process audio in frames to get more detailed feature variation
        feature_values = []
        hop_length = int(sample_rate / self.frame_rate)

        # Process audio in frames
        for i in range(self.frame_count):
            # Get frame of audio
            frame_start = min(int(i * hop_length), waveform.shape[-1] - 1)
            frame_end = min(frame_start + hop_length, waveform.shape[-1])

            if frame_start >= frame_end:
                # Use last valid frame if we're at the end
                if i > 0 and feature_values:
                    feature_values.append(feature_values[-1])
                else:
                    feature_values.append(0.0)  # Default
                continue

            # Extract frame - CRITICAL FIX: need to extract just this segment
            if waveform.dim() == 3:  # [batch, channels, samples]
                frame = waveform[
                    0, :, frame_start:frame_end
                ]  # First batch, all channels
            elif waveform.dim() == 2:  # [channels, samples]
                frame = waveform[:, frame_start:frame_end]  # All channels
            else:  # [samples]
                frame = waveform[frame_start:frame_end]  # Just samples

            # If we have a multi-channel audio, convert to mono
            if frame.dim() > 1:
                frame = torch.mean(frame, dim=0)

            # Skip empty frames
            if frame.numel() == 0:
                if i > 0 and feature_values:
                    feature_values.append(feature_values[-1])
                else:
                    feature_values.append(0.0)
                continue

            # Ensure frame is proper shape
            if frame.dim() == 0:  # It's a scalar
                if i > 0 and feature_values:
                    feature_values.append(feature_values[-1])
                else:
                    feature_values.append(0.0)
                continue

            print(f"EMD Frame {i} shape: {frame.shape}, numel: {frame.numel()}")

            # Convert to numpy
            if isinstance(frame, torch.Tensor):
                frame = frame.cpu().numpy()

            # Make sure frame is clean and valid
            frame = np.nan_to_num(frame)
            if len(frame) < 10:  # Need at least a few samples for meaningful filtering
                print(f"EMD Frame {i} too short: {len(frame)} samples")
                if i > 0 and feature_values:
                    feature_values.append(feature_values[-1])
                else:
                    feature_values.append(0.0)
                continue

            try:
                # Use the following settings for filter design (in Hz):
                # Fast oscillations: 8-20 Hz bandpass
                # Medium oscillations: 3-8 Hz bandpass
                # Slow oscillations: 0.5-3 Hz bandpass

                # These specific frequency ranges target typical oscillation rates in audio
                if self.feature_type == "emd_fast":
                    # High frequency band (8-20 Hz)
                    lowcut = 8.0
                    highcut = 20.0
                elif self.feature_type == "emd_medium":
                    # Medium frequency band (3-8 Hz)
                    lowcut = 3.0
                    highcut = 8.0
                else:  # emd_slow
                    # Low frequency band (0.5-3 Hz)
                    lowcut = 0.5
                    highcut = 3.0

                # Convert to normalized frequency for butterworth filter
                nyquist = sample_rate / 2.0
                low = lowcut / nyquist
                high = highcut / nyquist

                # Use lower filter order for stability with short frames
                order = 2

                # Design bandpass filter
                b, a = butter(order, [low, high], btype="band")

                # Apply filter if frame is long enough
                if len(frame) > (2 * order + 2):  # Need at least 2*order+2 samples
                    # Apply filter
                    filtered = filtfilt(b, a, frame)

                    # Calculate energy
                    energy = np.mean(np.abs(filtered))

                    # Apply some mild non-linear scaling to enhance differences
                    energy = np.power(energy, 0.7) * 10.0  # Scale up for visibility

                    print(f"EMD Frame {i} filtered energy: {energy}")
                    feature_values.append(float(energy))
                else:
                    # Direct energy for very short frames
                    energy = np.mean(np.abs(frame))
                    print(
                        f"EMD Frame {i} direct energy: {energy} (frame too short for filtering)"
                    )
                    feature_values.append(float(energy))
            except Exception as e:
                print(f"Error in EMD processing: {e}")
                # Use previous value or 0.0 as fallback
                if i > 0 and feature_values:
                    feature_values.append(feature_values[-1])
                else:
                    feature_values.append(0.0)

        # Check if we have variation in our values
        if len(feature_values) > 1:
            min_val = min(feature_values)
            max_val = max(feature_values)
            print(f"EMD Feature value range: {min_val} to {max_val}")

            if abs(max_val - min_val) < 1e-6:
                print("WARNING: No variation in EMD feature values!")
            else:
                print(f"EMD variation detected: {max_val - min_val}")

        # Store raw features - no normalization
        self.features = {self.feature_name: feature_values}

        return self

    def _normalize_features(self):
        for feature_name, values in self.features.items():
            min_val = min(values)
            max_val = max(values)
            range_val = max_val - min_val

            if range_val == 0:
                # Avoid division by zero
                normalized = [0.5 for _ in values]
            else:
                normalized = [(v - min_val) / range_val for v in values]

            self.normalized_features[feature_name] = normalized
