from .feature_extractors import FeatureExtractorBase
from .features_audio import (
    AudioFeature,
    PitchFeature,
    PitchRange,
    BaseFeature,
    RhythmFeature,
    WaveletFeature,
    EMDFeature,
)
from ... import RyanOnTheInside
from ..audio.audio_nodes import AudioNodeBase
from ...tooltips import apply_tooltips

_category = f"{FeatureExtractorBase.CATEGORY}/Audio"


class AudioFeatureExtractorMixin:
    @classmethod
    def INPUT_TYPES(cls):
        parent_inputs = super().INPUT_TYPES()["required"]
        parent_inputs["frame_count"] = ("INT", {"default": 0, "min": 0})
        return {
            **super().INPUT_TYPES(),
            "required": {
                **parent_inputs,
                "audio": ("AUDIO",),
            },
        }

    def calculate_target_frame_count(self, audio, frame_rate, frame_count):
        """Calculate the target frame count based on audio length and specified frame count"""
        waveform = audio["waveform"]
        sample_rate = audio["sample_rate"]
        natural_frame_count = int((waveform.shape[-1] / sample_rate) * frame_rate)
        return frame_count if frame_count > 0 else natural_frame_count


@apply_tooltips
class AudioFeatureExtractor(AudioFeatureExtractorMixin, FeatureExtractorBase):
    @classmethod
    def INPUT_TYPES(cls):
        parent_inputs = super().INPUT_TYPES()["required"]
        parent_inputs["extraction_method"] = (AudioFeature.get_extraction_methods(),)
        return {
            "required": {
                **parent_inputs,
                "audio": ("AUDIO",),
            }
        }

    RETURN_TYPES = (
        "FEATURE",
        "INT",
    )
    RETURN_NAMES = (
        "feature",
        "frame_count",
    )
    FUNCTION = "extract_feature"
    CATEGORY = _category

    def extract_feature(
        self, audio, frame_rate, frame_count, width, height, extraction_method
    ):
        target_frame_count = self.calculate_target_frame_count(
            audio, frame_rate, frame_count
        )

        feature = AudioFeature(
            width=width,
            height=height,
            feature_name=extraction_method,
            audio=audio,
            frame_count=target_frame_count,
            frame_rate=frame_rate,
            feature_type=extraction_method,
        )
        feature.extract()
        return (feature, target_frame_count)


@apply_tooltips
class RhythmFeatureExtractor(AudioFeatureExtractorMixin, FeatureExtractorBase):
    @classmethod
    def INPUT_TYPES(cls):
        parent_inputs = super().INPUT_TYPES()["required"]
        parent_inputs["extraction_method"] = (RhythmFeature.get_extraction_methods(),)
        return {
            "required": {
                **parent_inputs,
                "audio": ("AUDIO",),
                "time_signature": (
                    "INT",
                    {"default": 4, "min": 1, "max": 12, "step": 1},
                ),
            },
        }

    RETURN_TYPES = ("FEATURE",)
    FUNCTION = "extract_feature"
    CATEGORY = _category

    def extract_feature(
        self,
        audio,
        extraction_method,
        time_signature,
        frame_rate,
        frame_count,
        width,
        height,
    ):
        target_frame_count = self.calculate_target_frame_count(
            audio, frame_rate, frame_count
        )

        feature = RhythmFeature(
            width=width,
            height=height,
            feature_name=extraction_method,
            audio=audio,
            frame_count=target_frame_count,
            frame_rate=frame_rate,
            feature_type=extraction_method,
            time_signature=time_signature,
        )
        feature.extract()
        return (feature,)


@apply_tooltips
class PitchFeatureExtractor(AudioFeatureExtractorMixin, FeatureExtractorBase):
    @classmethod
    def INPUT_TYPES(cls):
        parent_inputs = super().INPUT_TYPES()["required"]
        parent_inputs["extraction_method"] = (PitchFeature.get_extraction_methods(),)
        return {
            "required": {
                **parent_inputs,
                "audio": ("AUDIO",),
                "opt_crepe_model": (
                    ["none", "medium", "tiny", "small", "large", "full"],
                    {"default": "medium"},
                ),
            },
            "optional": {
                "opt_pitch_range_collections": ("PITCH_RANGE_COLLECTION",),
            },
        }

    RETURN_TYPES = ("FEATURE",)
    FUNCTION = "extract_feature"
    CATEGORY = _category

    def extract_feature(
        self,
        audio,
        frame_rate,
        frame_count,
        width,
        height,
        extraction_method,
        opt_pitch_range_collections=None,
        opt_crepe_model=None,
    ):
        if opt_pitch_range_collections is None:
            opt_pitch_range_collections = []

        target_frame_count = self.calculate_target_frame_count(
            audio, frame_rate, frame_count
        )

        feature = PitchFeature(
            width=width,
            height=height,
            feature_name=extraction_method,
            audio=audio,
            frame_count=target_frame_count,
            frame_rate=frame_rate,
            pitch_range_collections=opt_pitch_range_collections,
            feature_type=extraction_method,
            crepe_model=opt_crepe_model,
        )
        feature.extract()
        return (feature,)


class PitchAbstraction(RyanOnTheInside):
    CATEGORY = "RyanOnTheInside/FlexFeatures/Audio/Pitch"


@apply_tooltips
class PitchRangeNode(PitchAbstraction):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "min_pitch": (
                    "FLOAT",
                    {"default": 80.0, "min": 20.0, "max": 2000.0, "step": 1.0},
                ),
                "max_pitch": (
                    "FLOAT",
                    {"default": 400.0, "min": 20.0, "max": 2000.0, "step": 1.0},
                ),
            },
            "optional": {
                "previous_range_collection": ("PITCH_RANGE_COLLECTION",),
            },
        }

    RETURN_TYPES = ("PITCH_RANGE_COLLECTION",)
    FUNCTION = "create_pitch_range"
    CATEGORY = _category

    def create_pitch_range(self, min_pitch, max_pitch, previous_range_collection=None):
        pitch_range = PitchRange(min_pitch, max_pitch)
        pitch_range_collection = {
            "pitch_ranges": [pitch_range],
            "chord_only": False,
        }
        if previous_range_collection is None:
            collections = [pitch_range_collection]
        else:
            collections = previous_range_collection + [pitch_range_collection]
        return (collections,)


@apply_tooltips
class PitchRangePresetNode(PitchAbstraction):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "preset": (
                    [
                        "Bass",
                        "Baritone",
                        "Tenor",
                        "Alto",
                        "Mezzo-soprano",
                        "Soprano",
                        "Contralto",
                    ],
                )
            },
            "optional": {
                "previous_range_collection": ("PITCH_RANGE_COLLECTION",),
            },
        }

    RETURN_TYPES = ("PITCH_RANGE_COLLECTION",)
    FUNCTION = "create_pitch_range_preset"
    CATEGORY = _category

    def create_pitch_range_preset(self, preset, previous_range_collection=None):
        presets = {
            "Bass": (82.41, 196.00),  # E2 - G3
            "Baritone": (98.00, 247.94),  # G2 - B3
            "Tenor": (130.81, 349.23),  # C3 - F4
            "Contralto": (130.81, 349.23),  # C3 - F4
            "Alto": (174.61, 440.00),  # F3 - A4
            "Mezzo-soprano": (196.00, 523.25),  # G3 - C5
            "Soprano": (261.63, 1046.50),  # C4 - C6
        }

        min_pitch, max_pitch = presets.get(preset, (20.0, 2000.0))
        pitch_range = PitchRange(min_pitch, max_pitch)
        pitch_range_collection = {
            "pitch_ranges": [pitch_range],
            "chord_only": False,
        }
        if previous_range_collection is None:
            collections = [pitch_range_collection]
        else:
            collections = previous_range_collection + [pitch_range_collection]
        return (collections,)


@apply_tooltips
class PitchRangeByNoteNode(PitchAbstraction):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "chord_only": ("BOOLEAN", {"default": False}),
                "pitch_tolerance_percent": (
                    "FLOAT",
                    {"default": 100.0, "min": 0.0, "max": 100.0, "step": 0.1},
                ),
                "notes": ("STRING", {"multiline": False}),
            },
            "optional": {
                "previous_range_collection": ("PITCH_RANGE_COLLECTION",),
            },
        }

    RETURN_TYPES = ("PITCH_RANGE_COLLECTION",)
    FUNCTION = "create_note_pitch_ranges"
    CATEGORY = _category

    def create_note_pitch_ranges(
        self, chord_only, notes, pitch_tolerance_percent, previous_range_collection=None
    ):
        if not notes:
            raise ValueError("At least one note must be selected.")

        # Parse the 'notes' string into a list of MIDI note numbers
        selected_notes = [
            int(note.strip()) for note in notes.split(",") if note.strip().isdigit()
        ]

        if not selected_notes:
            raise ValueError("No valid notes found in the 'notes' field.")

        pitch_ranges = []
        for midi_note in selected_notes:
            frequency = self._midi_to_frequency(midi_note)
            tolerance = PitchFeature.calculate_tolerance(
                frequency, pitch_tolerance_percent
            )
            min_pitch = frequency - tolerance
            max_pitch = frequency + tolerance
            pitch_range = PitchRange(min_pitch, max_pitch)
            pitch_ranges.append(pitch_range)

        # Create a collection with the 'chord_only' attribute
        pitch_range_collection = {
            "pitch_ranges": pitch_ranges,
            "chord_only": chord_only,
        }

        # Combine with previous collections if provided
        if previous_range_collection is None:
            collections = [pitch_range_collection]
        else:
            collections = previous_range_collection + [pitch_range_collection]

        return (collections,)

    def _midi_to_frequency(self, midi_note):
        import librosa

        return librosa.midi_to_hz(midi_note)

    # The _calculate_tolerance method has been removed from here


@apply_tooltips
class WaveletFeatureExtractor(AudioFeatureExtractorMixin, FeatureExtractorBase):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "extraction_method": (
                    [
                        "wavelet_high_frequency",
                        "wavelet_mid_frequency",
                        "wavelet_low_frequency",
                    ],
                    {"default": "wavelet_high_frequency"},
                ),
                "frame_count": (
                    "INT",
                    {"default": 16, "min": 1, "max": 1024, "step": 1},
                ),
                "frame_rate": (
                    "FLOAT",
                    {"default": 30.0, "min": 0.01, "max": 1000.0, "step": 0.01},
                ),
                "width": ("INT", {"default": 512, "min": 64, "max": 4096, "step": 64}),
                "height": ("INT", {"default": 512, "min": 64, "max": 4096, "step": 64}),
            },
        }

    RETURN_TYPES = ("FEATURE",)
    FUNCTION = "extract_feature"
    CATEGORY = _category

    def extract_feature(
        self, audio, extraction_method, frame_count, frame_rate, width, height
    ):
        from .features_audio import WaveletFeature

        # Validate extraction_method
        valid_methods = [
            "wavelet_high_frequency",
            "wavelet_mid_frequency",
            "wavelet_low_frequency",
        ]
        if extraction_method not in valid_methods:
            print(
                f"Warning: Invalid extraction_method '{extraction_method}'. Using default."
            )
            extraction_method = "wavelet_high_frequency"

        # Validate frame_rate
        if frame_rate < 0.01:
            print(f"Warning: Invalid frame_rate {frame_rate}. Using default.")
            frame_rate = 30.0

        # Calculate target frame count
        target_frame_count = self.calculate_target_frame_count(
            audio, frame_rate, frame_count
        )

        feature_name = "wavelet_feature"
        feature = WaveletFeature(
            width=width,
            height=height,
            feature_name=feature_name,
            audio=audio,
            frame_count=target_frame_count,
            frame_rate=frame_rate,
            feature_type=extraction_method,
        ).extract()

        return (feature,)


class EMDFeatureExtractor(AudioFeatureExtractorMixin, FeatureExtractorBase):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "extraction_method": (
                    ["emd_fast", "emd_medium", "emd_slow"],
                    {"default": "emd_fast"},
                ),
                "frame_count": (
                    "INT",
                    {"default": 16, "min": 1, "max": 1024, "step": 1},
                ),
                "frame_rate": (
                    "FLOAT",
                    {"default": 30.0, "min": 0.01, "max": 1000.0, "step": 0.01},
                ),
                "width": ("INT", {"default": 512, "min": 64, "max": 4096, "step": 64}),
                "height": ("INT", {"default": 512, "min": 64, "max": 4096, "step": 64}),
                "num_imfs": ("INT", {"default": 3, "min": 1, "max": 10, "step": 1}),
            },
        }

    RETURN_TYPES = ("FEATURE",)
    FUNCTION = "extract_feature"
    CATEGORY = _category

    def extract_feature(
        self, audio, extraction_method, frame_count, frame_rate, width, height, num_imfs
    ):
        from .features_audio import EMDFeature

        # Validate extraction_method
        valid_methods = ["emd_fast", "emd_medium", "emd_slow"]
        if extraction_method not in valid_methods:
            print(
                f"Warning: Invalid extraction_method '{extraction_method}'. Using default."
            )
            extraction_method = "emd_fast"

        # Validate frame_rate
        if frame_rate < 0.01:
            print(f"Warning: Invalid frame_rate {frame_rate}. Using default.")
            frame_rate = 30.0

        # Calculate target frame count
        target_frame_count = self.calculate_target_frame_count(
            audio, frame_rate, frame_count
        )

        feature_name = "emd_feature"
        feature = EMDFeature(
            width=width,
            height=height,
            feature_name=feature_name,
            audio=audio,
            frame_count=target_frame_count,
            frame_rate=frame_rate,
            feature_type=extraction_method,
        ).extract()

        return (feature,)


@apply_tooltips
class MelodicRangeFeatureExtractor(AudioFeatureExtractorMixin, FeatureExtractorBase):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "extraction_method": (
                    [
                        "melodic_low_range",
                        "melodic_mid_range",
                        "melodic_high_range",
                        "melodic_full_range",
                    ],
                    {"default": "melodic_mid_range"},
                ),
                "frame_count": (
                    "INT",
                    {"default": 16, "min": 1, "max": 1024, "step": 1},
                ),
                "frame_rate": (
                    "FLOAT",
                    {"default": 30.0, "min": 0.01, "max": 1000.0, "step": 0.01},
                ),
                "width": ("INT", {"default": 512, "min": 64, "max": 4096, "step": 64}),
                "height": ("INT", {"default": 512, "min": 64, "max": 4096, "step": 64}),
                "n_mels": ("INT", {"default": 128, "min": 16, "max": 512, "step": 8}),
            },
        }

    RETURN_TYPES = ("FEATURE",)
    FUNCTION = "extract_feature"
    CATEGORY = _category

    def extract_feature(
        self, audio, extraction_method, frame_count, frame_rate, width, height, n_mels
    ):
        from .features_audio import MelodicRangeFeature

        # Validate extraction_method
        valid_methods = [
            "melodic_low_range",
            "melodic_mid_range",
            "melodic_high_range",
            "melodic_full_range",
        ]
        if extraction_method not in valid_methods:
            print(
                f"Warning: Invalid extraction_method '{extraction_method}'. Using default."
            )
            extraction_method = "melodic_mid_range"

        # Validate frame_rate
        if frame_rate < 0.01:
            print(f"Warning: Invalid frame_rate {frame_rate}. Using default.")
            frame_rate = 30.0

        # Calculate target frame count
        target_frame_count = self.calculate_target_frame_count(
            audio, frame_rate, frame_count
        )

        feature_name = "melodic_range_feature"
        feature = MelodicRangeFeature(
            width=width,
            height=height,
            feature_name=feature_name,
            audio=audio,
            frame_count=target_frame_count,
            frame_rate=frame_rate,
            feature_type=extraction_method,
            n_mels=n_mels,
        ).extract()

        return (feature,)


@apply_tooltips
class NoteEventsFeatureExtractor(AudioFeatureExtractorMixin, FeatureExtractorBase):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                # "extraction_method": (
                #     [
                #         "note_onsets",
                #         "note_pitches",
                #         "note_durations",
                #         "note_density",
                #         "note_activity",
                #     ],
                #     {"default": "note_activity"},
                # ),
                "frame_count": (
                    "INT",
                    {"default": 16, "min": 1, "max": 1024, "step": 1},
                ),
                "frame_rate": (
                    "FLOAT",
                    {"default": 30.0, "min": 0.01, "max": 1000.0, "step": 0.01},
                ),
                "width": ("INT", {"default": 512, "min": 64, "max": 4096, "step": 64}),
                "height": ("INT", {"default": 512, "min": 64, "max": 4096, "step": 64}),
                "onset_threshold": (
                    "FLOAT",
                    {"default": 0.5, "min": 0.1, "max": 1.0, "step": 0.01},
                ),
                "min_note_duration": (
                    "FLOAT",
                    {"default": 0.1, "min": 0.01, "max": 1.0, "step": 0.01},
                ),
            },
            "optional": {
                "pitch_min": (
                    "FLOAT",
                    {"default": 50.0, "min": 20.0, "max": 500.0, "step": 1.0},
                ),
                "pitch_max": (
                    "FLOAT",
                    {"default": 2000.0, "min": 500.0, "max": 8000.0, "step": 10.0},
                ),
                "polyphony_enabled": ("BOOLEAN", {"default": True}),
                "opt_crepe_model": (
                    ["none", "medium", "tiny", "small", "large", "full"],
                    {"default": "medium"},
                ),
            },
        }

    RETURN_TYPES = ("FEATURE", "FEATURE", "FEATURE", "FEATURE", "FEATURE", "STRING")
    RETURN_NAMES = (
        "note_onsets_feature",
        "note_pitches_feature",
        "note_durations_feature",
        "note_density_feature",
        "note_activity_feature",
        "notes_json",
    )
    FUNCTION = "extract_feature"
    CATEGORY = _category

    def extract_feature(
        self,
        audio,
        # extraction_method,
        frame_count,
        frame_rate,
        width,
        height,
        onset_threshold,
        min_note_duration,
        pitch_min=50.0,
        pitch_max=2000.0,
        polyphony_enabled=True,
        opt_crepe_model="medium",
    ):
        from .features_audio import NoteEventsFeature, BaseAudioFeature
        import json
        import numpy as np

        # Create a concrete feature class that extends BaseAudioFeature
        class NoteEventsAudioFeature(BaseAudioFeature):
            @classmethod
            def get_extraction_methods(cls):
                return ["feature"]

            def extract(self):
                # Features are already set, no extraction needed
                return self

            def get_feature_sequence(self, feature_name=None):
                # Use our single feature type when feature_name is None
                if feature_name is None:
                    feature_name = self.feature_name
                return self.features.get(feature_name, [])

            def get_value_at_frame(self, frame_index):
                # Override to use our feature_name directly
                feature_type = self.feature_name
                if feature_type in self.features:
                    return self.features[feature_type][frame_index]
                return 0.0

        # Validate frame_rate
        if frame_rate < 0.01:
            print(f"Warning: Invalid frame_rate {frame_rate}. Using default.")
            frame_rate = 30.0

        # Calculate target frame count
        target_frame_count = self.calculate_target_frame_count(
            audio, frame_rate, frame_count
        )

        # Create a single feature object that will extract all features at once
        feature = NoteEventsFeature(
            width=width,
            height=height,
            feature_name="note_events",
            audio=audio,
            frame_count=target_frame_count,
            frame_rate=frame_rate,
            feature_type="note_activity",  # Default type
            min_note_duration=min_note_duration,
            onset_threshold=onset_threshold,
            pitch_min=pitch_min,
            pitch_max=pitch_max,
            polyphony=polyphony_enabled,
            crepe_model=opt_crepe_model,
        ).extract()

        # Get the detected notes as JSON
        notes = feature.get_all_notes()

        # Convert numpy types to standard Python types for JSON serialization
        notes_json_safe = []
        for note in notes:
            json_safe_note = {}
            for key, value in note.items():
                # Convert numpy numeric types to Python float or int
                if isinstance(
                    value, (np.integer, np.int_, np.int8, np.int16, np.int32, np.int64)
                ):
                    json_safe_note[key] = int(value)
                elif isinstance(
                    value, (np.floating, np.float_, np.float16, np.float32, np.float64)
                ):
                    json_safe_note[key] = float(value)
                else:
                    json_safe_note[key] = value
            notes_json_safe.append(json_safe_note)

        # Now serialize to JSON
        notes_json = json.dumps(notes_json_safe, indent=2)

        # Create 5 separate feature objects - one for each feature type
        # Each one will contain only its specific feature data

        # 1. Note Onsets Feature
        onsets = NoteEventsAudioFeature(
            "note_onsets", audio, target_frame_count, frame_rate, width, height
        )
        onsets.features = {"note_onsets": feature.features["note_onsets"]}
        onsets.feature_name = "note_onsets"  # Ensure feature_name matches the key

        # 2. Note Pitches Feature
        pitches = NoteEventsAudioFeature(
            "note_pitches", audio, target_frame_count, frame_rate, width, height
        )
        pitches.features = {"note_pitches": feature.features["note_pitches"]}
        pitches.feature_name = "note_pitches"  # Ensure feature_name matches the key

        # 3. Note Durations Feature
        durations = NoteEventsAudioFeature(
            "note_durations", audio, target_frame_count, frame_rate, width, height
        )
        durations.features = {"note_durations": feature.features["note_durations"]}
        durations.feature_name = "note_durations"  # Ensure feature_name matches the key

        # 4. Note Density Feature
        density = NoteEventsAudioFeature(
            "note_density", audio, target_frame_count, frame_rate, width, height
        )
        density.features = {"note_density": feature.features["note_density"]}
        density.feature_name = "note_density"  # Ensure feature_name matches the key

        # 5. Note Activity Feature
        activity = NoteEventsAudioFeature(
            "note_activity", audio, target_frame_count, frame_rate, width, height
        )
        activity.features = {"note_activity": feature.features["note_activity"]}
        activity.feature_name = "note_activity"  # Ensure feature_name matches the key

        return (
            onsets,
            pitches,
            durations,
            density,
            activity,
            notes_json,
        )
