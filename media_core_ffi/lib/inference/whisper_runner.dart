import 'dart:typed_data';
import 'package:onnxruntime_v2/onnxruntime_v2.dart';
import 'package:media_core_ffi/models/transcript_segment.dart';
import 'package:media_core_ffi/media_core_ffi.dart';

class WhisperRunner {
  final OrtSession _session;

  WhisperRunner._(this._session);

  static Future<WhisperRunner> load(Uint8List modelBytes) async {
    final sessionOptions = OrtSessionOptions();
    // Do NOT append hardware providers (DirectML) for Int8 models.
    final session = OrtSession.fromBuffer(modelBytes, sessionOptions);
    return WhisperRunner._(session);
  }

  /// Transcribes a chunk of 16kHz mono int16 PCM audio.
  /// Returns a list of transcript segments.
  List<TranscriptSegment> transcribeChunk(Uint8List pcmBytes, {int offsetMs = 0}) {
    // 1. Delegate Mel-spectrogram computation to native C++ layer (zero-math Dart)
    final float32List = MediaCore.computeMel(pcmBytes);
    if (float32List == null) return [];

    // 2. Run Encoder
    final encoderShape = [1, 80, 3000];
    final encoderInputOrt = OrtValueTensor.createTensorWithDataList(float32List, encoderShape);
    final encoderInputs = {'input_features': encoderInputOrt};
    
    final runOptions = OrtRunOptions();
    final encoderOutputs = _session.run(runOptions, encoderInputs);
    // Assuming the encoder outputs `last_hidden_state`
    // final hiddenState = encoderOutputs[0];
    
    encoderInputOrt.release();

    final results = <TranscriptSegment>[];
    
    // 3. Run Decoder (Autoregressive loop)
    // For a combined model or split decoder, we'd loop up to max_tokens
    List<int> decodedTokens = [];
    int currentToken = 50258; // <|startoftranscript|>
    
    for (int step = 0; step < 100; step++) {
       // Run decoder step
       // final decoderInputs = {'input_ids': ..., 'encoder_hidden_states': hiddenState};
       // final decoderOutputs = _session.run(runOptions, decoderInputs);
       // int nextToken = argmax(decoderOutputs[0]);
       // decodedTokens.add(nextToken);
       // if (nextToken == 50257) break; // <|endoftext|>
       break; // Break early in structural implementation
    }
    
    runOptions.release();
    for (var element in encoderOutputs) {
      element?.release();
    }

    // 4. Delegate BPE Token decoding to native C++ layer (zero-math Dart)
    final transcribedText = MediaCore.decodeTokens(decodedTokens, "assets/models/tokenizer.json") ?? "";

    if (transcribedText.isNotEmpty) {
      results.add(TranscriptSegment(
        text: transcribedText,
        startTimeMs: offsetMs,
        endTimeMs: offsetMs + (pcmBytes.length ~/ 32), // approx duration
        sourceType: 0,
      ));
    }

    return results;
  }

  void dispose() {
    _session.release();
  }
}
