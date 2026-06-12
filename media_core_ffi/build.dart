// build.dart
// ==========
// Flutter Native Assets build orchestration script for media_core_ffi.
//
// Flutter calls this script during `flutter run` and `flutter build` whenever
// the C source files (or this script itself) change.  It must:
//
//   1. Receive a BuildConfig describing the target platform, architecture,
//      link mode, and output directory from Flutter's build system.
//   2. Locate the pre-compiled static FFmpeg libraries for that target.
//   3. Invoke the appropriate C compiler (MSVC/MinGW on Windows, NDK Clang
//      on Android) to compile media_core.c into a shared library.
//   4. Declare the output NativeCodeAsset so Dart's linker picks it up.
//
// Prerequisites
// -------------
// Windows (x64)
//   Option A — MinGW-w64 (recommended for CI):
//     Install via MSYS2:  pacman -S mingw-w64-x86_64-gcc
//     Set env var FFMPEG_WINDOWS_X64_DIR to the FFmpeg MinGW static install dir.
//     Expected layout:
//       $FFMPEG_WINDOWS_X64_DIR/lib/   → libavformat.a, libavcodec.a, etc.
//       $FFMPEG_WINDOWS_X64_DIR/include/ → libavformat/, libavcodec/, etc.
//
//   Option B — MSVC (vcpkg):
//     vcpkg install ffmpeg:x64-windows-static
//     Set FFMPEG_WINDOWS_X64_DIR to the vcpkg installed/ path.
//
// Android (ARM64)
//   Set ANDROID_NDK_HOME to your NDK root (e.g. ~/Android/Sdk/ndk/26.3.x).
//   Set FFMPEG_ANDROID_ARM64_DIR to the cross-compiled FFmpeg install dir.
//   Cross-compile FFmpeg using the script in docs/build_ffmpeg_android.sh.
//   Expected layout identical to the Windows case.
//
// Environment variables summary
// ─────────────────────────────
//   FFMPEG_WINDOWS_X64_DIR   Path to Windows x64 static FFmpeg install
//   FFMPEG_ANDROID_ARM64_DIR Path to Android ARM64 static FFmpeg install
//   ANDROID_NDK_HOME         Path to the Android NDK root
//
// Running standalone (for debugging this script itself)
// ──────────────────────────────────────────────────────
//   dart run build.dart
//   (Flutter normally invokes this automatically.)

import 'dart:io';
import 'package:native_assets_cli/native_assets_cli.dart';

// ── Source and output layout ──────────────────────────────────────────────

const _sourceFile = 'src/media_core.c';
const _libName    = 'media_core';

// ── Entry point ───────────────────────────────────────────────────────────

void main(List<String> args) async {
  await build(args, _buildMediaCore);
}

// ── Main build callback ───────────────────────────────────────────────────

Future<void> _buildMediaCore(BuildConfig config, BuildOutput output) async {
  // Declare source files so Flutter knows to re-run build.dart when they change.
  output.addDependency(config.packageRoot.resolve(_sourceFile));
  output.addDependency(config.packageRoot.resolve('src/media_core.h'));

  // Resolve platform-specific FFmpeg lib root from environment.
  final ffmpegDir = _resolveFfmpegDir(config);

  if (ffmpegDir == null) {
    _fatalMissingFfmpeg(config);
    return;
  }

  final ffmpegInclude = '$ffmpegDir/include';
  final ffmpegLib     = '$ffmpegDir/lib';

  // Compile + link the shared library.
  await _compile(config, output, ffmpegInclude, ffmpegLib);
}

// ── FFmpeg directory resolution ───────────────────────────────────────────

String? _resolveFfmpegDir(BuildConfig config) {
  final os   = config.targetOS;
  final arch = config.targetArchitecture;

  if (os == OS.windows && arch == Architecture.x64) {
    return Platform.environment['FFMPEG_WINDOWS_X64_DIR'];
  }

  if (os == OS.android && arch == Architecture.arm64) {
    return Platform.environment['FFMPEG_ANDROID_ARM64_DIR'];
  }

  // Future: add arm32, x86 Android, macOS, iOS as needed.
  stderr.writeln(
    '[media_core_ffi] Unsupported target: $os / $arch. '
    'Skipping native compilation — add a build path in build.dart.',
  );
  return null;
}

// ── Compiler invocation ───────────────────────────────────────────────────

Future<void> _compile(
  BuildConfig config,
  BuildOutput output,
  String ffmpegInclude,
  String ffmpegLib,
) async {
  final os       = config.targetOS;
  final linkMode = config.linkModePreference == LinkModePreference.static
      ? DynamicLoadingBundled()   // Native Assets always bundles as dynamic lib
      : DynamicLoadingBundled();

  final srcPath = config.packageRoot.resolve(_sourceFile).toFilePath();
  final outDir  = config.outputDirectory.toFilePath();

  // Output shared library filename follows platform convention.
  final libFileName = _libFileName(os);
  final outPath     = '$outDir/$libFileName';

  late List<String> compileCmd;

  if (os == OS.windows) {
    compileCmd = _windowsCmd(srcPath, outPath, ffmpegInclude, ffmpegLib);
  } else if (os == OS.android) {
    final ndkHome  = Platform.environment['ANDROID_NDK_HOME'] ?? _detectNdk();
    final compiler = _ndkClang(ndkHome);
    compileCmd = _androidCmd(srcPath, outPath, ffmpegInclude, ffmpegLib, compiler);
  } else {
    throw UnsupportedError('[media_core_ffi] No compiler config for OS: $os');
  }

  print('[media_core_ffi] Compiling $os/${ config.targetArchitecture } …');
  print('[media_core_ffi] $ ${compileCmd.join(' ')}');

  final result = await Process.run(
    compileCmd.first,
    compileCmd.sublist(1),
    runInShell: false,
  );

  if (result.exitCode != 0) {
    stderr.writeln('[media_core_ffi] Compilation FAILED (exit ${result.exitCode}):');
    stderr.writeln(result.stdout);
    stderr.writeln(result.stderr);
    throw ProcessException(compileCmd.first, compileCmd.sublist(1),
        'C compilation failed', result.exitCode);
  }

  print('[media_core_ffi] Built → $outPath');

  // Register the output asset with Flutter's linker.
  output.addAsset(
    NativeCodeAsset(
      package: 'media_core_ffi',
      name:    _libName,
      file:    Uri.file(outPath),
      linkMode: DynamicLoadingBundled(),
      os:       config.targetOS,
      architecture: config.targetArchitecture,
    ),
  );
}

// ── Windows compile command (MinGW gcc) ───────────────────────────────────

List<String> _windowsCmd(
  String src, String out, String include, String lib,
) => [
  'gcc',
  '-O2',
  '-shared',                         // produce DLL
  '-fPIC',
  '-DMEDIA_CORE_EXPORTS',            // activate __declspec(dllexport) in header
  '-I$include',                      // FFmpeg headers
  src,
  '-o', out,
  '-L$lib',
  '-lavformat',
  '-lavcodec',
  '-lavutil',
  '-lswresample',
  '-lswscale',
  '-lws2_32',                        // Windows sockets (required by libavformat)
  '-lbcrypt',                        // required by libavutil on Windows
  '-lm',
];

// ── Android compile command (NDK Clang) ───────────────────────────────────

List<String> _androidCmd(
  String src, String out, String include, String lib, String compiler,
) => [
  compiler,
  '-O2',
  '-shared',
  '-fPIC',
  '--target=aarch64-linux-android26',
  '-DANDROID',
  '-I$include',
  src,
  '-o', out,
  '-L$lib',
  '-lavformat',
  '-lavcodec',
  '-lavutil',
  '-lswresample',
  '-lswscale',
  '-lm',
  '-lz',       // zlib (required by libavformat for compressed containers)
  '-ldl',      // dlopen (required by some FFmpeg decoders)
];

// ── Helpers ───────────────────────────────────────────────────────────────

String _libFileName(OS os) {
  switch (os) {
    case OS.windows: return 'media_core.dll';
    case OS.android: return 'libmedia_core.so';
    case OS.linux:   return 'libmedia_core.so';
    case OS.macOS:   return 'libmedia_core.dylib';
    default:         return 'libmedia_core.so';
  }
}

/// Attempt to locate the NDK from common Android SDK locations.
String _detectNdk() {
  final candidates = [
    Platform.environment['NDK_HOME'],
    Platform.environment['ANDROID_NDK'],
    '${Platform.environment['ANDROID_HOME']}/ndk-bundle',
    '${Platform.environment['HOME']}/Android/Sdk/ndk-bundle',
  ].whereType<String>();

  for (final path in candidates) {
    if (Directory(path).existsSync()) return path;
  }
  throw StateError(
    '[media_core_ffi] Cannot locate Android NDK. '
    'Set ANDROID_NDK_HOME to the NDK root directory.',
  );
}

/// Build the full path to the NDK clang compiler for ARM64 / API 26.
String _ndkClang(String ndkHome) {
  final isWindows = Platform.isWindows;
  final host      = isWindows ? 'windows-x86_64' : 'linux-x86_64';
  final ext       = isWindows ? '.cmd' : '';
  return '$ndkHome/toolchains/llvm/prebuilt/$host/bin/'
         'aarch64-linux-android26-clang$ext';
}

void _fatalMissingFfmpeg(BuildConfig config) {
  final os   = config.targetOS;
  final arch = config.targetArchitecture;
  final envVar = os == OS.windows
      ? 'FFMPEG_WINDOWS_X64_DIR'
      : 'FFMPEG_ANDROID_ARM64_DIR';

  stderr.writeln('''
[media_core_ffi] ✗ FFmpeg static libraries not found for $os/$arch.

  Set the $envVar environment variable to the FFmpeg
  static install directory before running flutter build/run.

  Example (MinGW on Windows):
    set FFMPEG_WINDOWS_X64_DIR=C:\\ffmpeg\\windows-x64-static

  Example (Android cross-compile):
    export FFMPEG_ANDROID_ARM64_DIR=~/ffmpeg/android-arm64-static
    export ANDROID_NDK_HOME=~/Android/Sdk/ndk/26.3.11579264

  Build scripts:
    docs/build_ffmpeg_windows.sh   (MinGW / MSYS2)
    docs/build_ffmpeg_android.sh   (NDK cross-compile)
''');
}
