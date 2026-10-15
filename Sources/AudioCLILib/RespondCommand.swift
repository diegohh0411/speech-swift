import Foundation
import ArgumentParser
import PersonaPlex
import AudioCommon
import MLX

public struct RespondCommand: ParsableCommand {
    public static let configuration = CommandConfiguration(
        commandName: "respond",
        abstract: "Full-duplex speech-to-speech inference using PersonaPlex 7B"
    )

    @Option(name: .shortAndLong, help: "Input audio WAV file (24kHz mono)")
    public var input: String?

    @Option(name: .shortAndLong, help: "Output response WAV file")
    public var output: String = "response.wav"

    @Option(name: .long, help: "Voice preset (e.g. NATM0, NATF1, VARF0)")
    public var voice: String = "NATM0"

    @Option(name: .long, help: "System prompt preset: assistant, focused, customer-service, teacher")
    public var systemPrompt: String = "assistant"

    @Option(name: .long, help: "Custom system prompt text (overrides --system-prompt preset)")
    public var systemPromptText: String?

    @Option(name: .long, help: "Maximum generation steps at 12.5Hz (default: 200 = ~16s)")
    public var maxSteps: Int = 200

    @Option(name: .long, help: "HuggingFace model ID (e.g. aufklarer/PersonaPlex-7B-MLX-4bit or aufklarer/PersonaPlex-7B-MLX-8bit)")
    public var modelId: String = PersonaPlexModel.defaultModelId

    @Flag(name: .long, help: "Enable streaming output (emit audio chunks during generation)")
    public var stream: Bool = false

    @Option(name: .long, help: "Frames per streaming chunk (default: 25 = ~2s)")
    public var chunkFrames: Int = 25

    @Flag(name: .long, help: "Enable compiled temporal transformer (warmup + kernel fusion)")
    public var compile: Bool = false

    @Flag(name: .long, help: "List available voices and exit")
    public var listVoices: Bool = false

    @Flag(name: .long, help: "List available system prompt presets and exit")
    public var listPrompts: Bool = false

    @Flag(name: .long, help: "Show detailed timing info")
    public var verbose: Bool = false

    @Flag(name: .long, help: "Decode and print the model's inner monologue text")
    public var transcript: Bool = false

    @Flag(name: .long, help: "Output results as JSON (includes transcript, latency, audio path)")
    public var json: Bool = false

    // Sampling config overrides
    @Option(name: .long, help: "Audio sampling temperature (default: 0.8)")
    public var audioTemp: Float?

    @Option(name: .long, help: "Text sampling temperature (default: 0.7)")
    public var textTemp: Float?

    @Option(name: .long, help: "Audio top-k candidates (default: 250)")
    public var audioTopK: Int?

    @Option(name: .long, help: "Audio repetition penalty (default: 1.2, 1.0 = disabled)")
    public var repetitionPenalty: Float?

    @Option(name: .long, help: "Text repetition penalty (default: 1.2, 1.0 = disabled)")
    public var textRepetitionPenalty: Float?

    @Option(name: .long, help: "Repetition penalty window in frames (default: 30)")
    public var repetitionWindow: Int?

    @Option(name: .long, help: "Silence frames before early stop (default: 15, 0 = disabled)")
    public var silenceEarlyStop: Int?

    @Option(name: .long, help: "Text entropy threshold for early stop (default: 0 = disabled, try 1.0)")
    public var entropyThreshold: Float?

    @Option(name: .long, help: "Consecutive low-entropy steps before early stop (default: 10)")
    public var entropyWindow: Int?

    @Flag(name: .long, help: "Full-duplex mode: feed input WAV into ring buffer, record response. Saves input + output WAVs for offline analysis.")
    public var fullDuplex: Bool = false

    @Option(name: .long, help: "Full-duplex debug output directory (default: /tmp/personaplex_debug)")
    public var debugDir: String = "/tmp/personaplex_debug"

    @Option(name: .long, help: "Full-duplex: text to force into the agent's text stream")
    public var injectText: String?

    @Option(name: .long, help: "Full-duplex: generation step at which to enqueue --inject-text")
    public var injectAt: Int = 100

    @Option(name: .long, help: "Full-duplex: padding tokens forced after each injected token")
    public var injectPads: Int = 1

    @Option(name: .long, help: "Full-duplex: OpenAI-compatible base URL of a director LLM (e.g. http://localhost:9932)")
    public var directorURL: String?

    @Option(name: .long, help: "Full-duplex: what the student said (stand-in for ASR), sent to the director")
    public var studentText: String = ""

    @Option(name: .long, help: "Full-duplex: director system prompt")
    public var directorPrompt: String =
        "You write the exact words a friendly math tutor says out loud next. Rules: at most two short sentences, under 30 words total. Give one small hint, then ask the student one guiding question. Never state the final answer. Spoken words only: no symbols, no formulas, no markdown."

    @Option(name: .long, help: "Full-duplex: padding steps forced after the director reply (turn hold)")
    public var holdSteps: Int = 0

    @Flag(name: .long, help: "Full-duplex: force padding while waiting for the director")
    public var thinkPads: Bool = false

    enum DirectorState { case idle, thinking, speaking }

    final class DirectorBox: @unchecked Sendable {
        private let lock = NSLock()
        private var value: (String, Double)?
        func set(_ s: String, latency: Double) { lock.lock(); value = (s, latency); lock.unlock() }
        func get() -> (String, Double)? { lock.lock(); defer { lock.unlock() }; return value }
    }

    static func askDirector(url: String, system: String, student: String) async throws -> String {
        var req = URLRequest(url: URL(string: url)!.appendingPathComponent("v1/chat/completions"))
        req.httpMethod = "POST"
        req.setValue("application/json", forHTTPHeaderField: "Content-Type")
        let body: [String: Any] = [
            // LFM2.5 always reasons; an empty think block as assistant prefill skips it (~0.7 s replies).
            "messages": [
                ["role": "system", "content": system],
                ["role": "user", "content": "Student says: \"\(student)\"\nTutor says:"],
                ["role": "assistant", "content": "<think>\n</think>\n"],
            ],
            "temperature": 0.3, "max_tokens": 60,
        ]
        req.httpBody = try JSONSerialization.data(withJSONObject: body)
        let (data, _) = try await URLSession.shared.data(for: req)
        let json = try JSONSerialization.jsonObject(with: data) as? [String: Any]
        let msg = (json?["choices"] as? [[String: Any]])?.first?["message"] as? [String: Any]
        var text = (msg?["content"] as? String) ?? ""
        if let r = text.range(of: "</think>") { text = String(text[r.upperBound...]) }
        return text.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    public init() {}

    public func run() throws {
        if listVoices {
            print("Available voices:")
            for v in PersonaPlexVoice.allCases {
                print("  \(v.rawValue) - \(v.displayName)")
            }
            return
        }

        if listPrompts {
            print("Available system prompts:")
            for p in SystemPromptPreset.allCases {
                print("  \(p.rawValue) - \(p.description)")
            }
            return
        }

        guard let input = input else {
            print("Error: --input is required for inference.")
            throw ExitCode(1)
        }

        guard let selectedVoice = PersonaPlexVoice(rawValue: voice) else {
            print("Unknown voice: \(voice)")
            print("Use --list-voices to see available options.")
            throw ExitCode(1)
        }

        var selectedPrompt: SystemPromptPreset?
        if systemPromptText == nil {
            guard let preset = SystemPromptPreset(rawValue: systemPrompt) else {
                print("Unknown system prompt: \(systemPrompt)")
                print("Use --list-prompts to see available options.")
                throw ExitCode(1)
            }
            selectedPrompt = preset
        }

        try runAsync {
            if verbose {
                print("Build: \(buildVersion)")
            }
            print("Loading PersonaPlex 7B model...")
            let model = try await PersonaPlexModel.fromPretrained(
                modelId: modelId, progressHandler: reportProgress)

            // Cap MLX's buffer cache (default 1 GB); uncapped, steps get ~8x slower.
            let cacheMB = ProcessInfo.processInfo.environment["PP_CACHE_MB"].flatMap { Int($0) } ?? 1024
            MLX.Memory.cacheLimit = cacheMB * 1024 * 1024
            print("MLX cacheLimit = \(cacheMB) MB")

            // Apply sampling config overrides
            if let v = audioTemp { model.cfg.sampling.audioTemp = v }
            if let v = textTemp { model.cfg.sampling.textTemp = v }
            if let v = audioTopK { model.cfg.sampling.audioTopK = v }
            if let v = repetitionPenalty { model.cfg.sampling.audioRepetitionPenalty = v }
            if let v = textRepetitionPenalty { model.cfg.sampling.textRepetitionPenalty = v }
            if let v = repetitionWindow { model.cfg.sampling.repetitionWindow = v }
            if let v = silenceEarlyStop { model.cfg.sampling.silenceEarlyStopFrames = v }
            if let v = entropyThreshold { model.cfg.sampling.entropyEarlyStopThreshold = v }
            if let v = entropyWindow { model.cfg.sampling.entropyWindow = v }

            if compile {
                print("Warming up compiled temporal transformer...")
                let warmStart = CFAbsoluteTimeGetCurrent()
                model.warmUp()
                let warmTime = CFAbsoluteTimeGetCurrent() - warmStart
                print("  Warmup: \(String(format: "%.2f", warmTime))s")
            }

            // Use the model's built-in tokenizer for transcript decoding
            let spmDecoder = model.tokenizer

            // Resolve system prompt tokens
            let promptTokens: [Int32]?
            let promptLabel: String
            if let customText = systemPromptText {
                guard let tokens = model.tokenizeSystemPrompt(customText) else {
                    print("Error: SentencePiece tokenizer not available for custom system prompt.")
                    throw ExitCode(1)
                }
                promptTokens = tokens
                promptLabel = "custom (\(tokens.count) tokens)"
            } else {
                promptTokens = selectedPrompt?.tokens
                promptLabel = selectedPrompt?.rawValue ?? "default"
            }

            if !json { print("Loading input audio: \(input)") }
            let inputURL = URL(fileURLWithPath: input)
            let audio = try AudioFileLoader.load(
                url: inputURL, targetSampleRate: 24000)
            let duration = Double(audio.count) / 24000.0
            if !json {
                print("  Duration: \(String(format: "%.2f", duration))s (\(audio.count) samples)")
                print("Generating response with voice \(selectedVoice.rawValue), prompt: \(promptLabel)")
            }
            let startTime = CFAbsoluteTimeGetCurrent()

            var responseSamples: [Float] = []
            var textTokens: [Int32] = []

            if fullDuplex {
                // Full-duplex debug mode: feed input into ring buffer, record all output
                let dir = URL(fileURLWithPath: debugDir)
                try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)

                print("Full-duplex mode: feeding \(String(format: "%.1f", duration))s input...")
                let ringBuffer = AudioRingBuffer(capacity: 24000 * 10)

                // Feed input audio into ring buffer in real-time-ish chunks
                let chunkSize = 1920  // 80ms at 24kHz (one Mimi frame)
                Task.detached {
                    var offset = 0
                    while offset < audio.count {
                        let end = min(offset + chunkSize, audio.count)
                        let chunk = Array(audio[offset..<end])
                        ringBuffer.write(chunk)
                        offset = end
                        // Simulate real-time: 80ms per chunk
                        try? await Task.sleep(nanoseconds: 80_000_000)
                    }
                    print("  Input feed complete (\(audio.count) samples)")
                }

                let injector = TextTokenInjector()
                var injectTokens: [Int32] = []
                if let text = injectText, let dec = spmDecoder {
                    for tok in dec.encode(text) {
                        injectTokens.append(tok)
                        injectTokens += [Int32](repeating: 3, count: injectPads)
                    }
                    print("Will inject \(injectTokens.count) tokens at step \(injectAt): \(dec.encode(text))")
                }
                let stream = model.respondRealtimeFrames(
                    voice: selectedVoice,
                    systemPromptTokens: promptTokens,
                    userAudioBuffer: ringBuffer,
                    maxSteps: maxSteps,
                    textInjector: injector,
                    verbose: verbose
                )

                var directorState: DirectorState = .idle
                var directorReply = ""
                let director = DirectorBox()
                var frameCount = 0
                var stepTimes: [Double] = []
                var lastFrameTime = CFAbsoluteTimeGetCurrent()
                var segments: [(forced: Bool, tokens: [Int32])] = []
                var tokenLog = ""
                for try await frame in stream {
                    let now = CFAbsoluteTimeGetCurrent()
                    if frameCount > 0 { stepTimes.append((now - lastFrameTime) * 1000) }
                    lastFrameTime = now
                    if frameCount == injectAt && !injectTokens.isEmpty {
                        injector.enqueue(injectTokens)  // pushed from outside the loop, like a director would
                    }
                    if frameCount == injectAt, let url = directorURL {
                        // Ask the director LLM for the tutor's reply; force pads while it thinks.
                        directorState = .thinking
                        let reqStart = CFAbsoluteTimeGetCurrent()
                        let student = studentText
                        let sys = directorPrompt
                        let box = director
                        Task.detached {
                            let reply = (try? await Self.askDirector(url: url, system: sys, student: student)) ?? ""
                            box.set(reply, latency: CFAbsoluteTimeGetCurrent() - reqStart)
                        }
                    }
                    if directorState == .thinking {
                        if let (reply, latency) = director.get() {
                            print(String(format: "Director reply after %.2fs (step %d): %@", latency, frameCount, reply))
                            directorReply = reply
                            injector.clear()
                            var toks: [Int32] = []
                            for tok in spmDecoder!.encode(reply) {
                                toks.append(tok)
                                toks += [Int32](repeating: 3, count: injectPads)
                            }
                            toks += [Int32](repeating: 3, count: holdSteps)
                            injector.enqueue(toks)
                            directorState = .speaking
                        } else if thinkPads && injector.pending == 0 {
                            injector.enqueue([3])
                        }
                    }
                    responseSamples.append(contentsOf: frame.samples)
                    textTokens.append(frame.textToken)
                    if let last = segments.last, last.forced == frame.forced {
                        segments[segments.count - 1].tokens.append(frame.textToken)
                    } else {
                        segments.append((frame.forced, [frame.textToken]))
                    }
                    let piece = frame.textToken == 3 ? "" : (spmDecoder?.decode([frame.textToken]) ?? "")
                    let flag = frame.forced ? "F" : ""
                    let rms = (frame.samples.reduce(0) { $0 + $1 * $1 } / Float(max(frame.samples.count, 1))).squareRoot()
                    tokenLog += "\(frame.step)\t\(frame.textToken)\t\(flag)\t\(String(format: "%.4f", rms))\t\(piece)\n"
                    frameCount += 1
                    if frameCount % 25 == 0 {
                        let elapsed = CFAbsoluteTimeGetCurrent() - startTime
                        let audioSec = Double(responseSamples.count) / 24000.0
                        print("  Frame \(frameCount): \(String(format: "%.1f", audioSec))s audio, \(String(format: "%.1f", elapsed))s elapsed")
                    }
                }

                if let dec = spmDecoder {
                    var t = ""
                    var stepAt = 0
                    for seg in segments {
                        let txt = dec.decode(seg.tokens.filter { $0 != 3 })
                        let ts = String(format: "%.1fs", Double(stepAt) * 0.08)
                        t += seg.forced ? " [FORCED@\(ts): \(txt)]" : " {\(ts)} \(txt)"
                        stepAt += seg.tokens.count
                    }
                    print("Transcript:\(t)")
                }
                let sorted = stepTimes.sorted()
                if !sorted.isEmpty {
                    let mean = sorted.reduce(0, +) / Double(sorted.count)
                    print(String(format: "Step time: mean %.1f ms, p50 %.1f, p95 %.1f, max %.1f (n=%d)",
                        mean, sorted[sorted.count / 2], sorted[Int(Double(sorted.count) * 0.95)],
                        sorted.last!, sorted.count))
                }
                try tokenLog.write(
                    to: dir.appendingPathComponent("tokens.tsv"), atomically: true, encoding: .utf8)

                let elapsed = CFAbsoluteTimeGetCurrent() - startTime
                let outDuration = Double(responseSamples.count) / 24000.0
                print("Full-duplex complete: \(frameCount) frames, \(String(format: "%.1f", outDuration))s audio in \(String(format: "%.1f", elapsed))s")

                // Save debug files
                let inputPath = dir.appendingPathComponent("input.wav")
                let outputPath = dir.appendingPathComponent("output.wav")
                try WAVWriter.write(samples: audio, sampleRate: 24000, to: inputPath)
                try WAVWriter.write(samples: responseSamples, sampleRate: 24000, to: outputPath)
                print("Saved: \(inputPath.path)")
                print("Saved: \(outputPath.path)")
                return
            } else if stream {
                let streamingConfig = PersonaPlexModel.PersonaPlexStreamingConfig(
                    firstChunkFrames: chunkFrames, chunkFrames: chunkFrames)
                let audioStream = model.respondStream(
                    userAudio: audio,
                    voice: selectedVoice,
                    systemPromptTokens: promptTokens,
                    maxSteps: maxSteps,
                    streaming: streamingConfig,
                    verbose: verbose)

                var chunkCount = 0
                for try await chunk in audioStream {
                    responseSamples.append(contentsOf: chunk.samples)
                    chunkCount += 1
                    if chunk.isFinal { textTokens = chunk.textTokens }
                    if verbose {
                        let chunkDuration = Double(chunk.samples.count) / 24000.0
                        print("  Chunk \(chunkCount): \(chunk.samples.count) samples (\(String(format: "%.2f", chunkDuration))s) at \(String(format: "%.2f", chunk.elapsedTime ?? 0))s")
                    }
                }
            } else {
                let result = model.respond(
                    userAudio: audio,
                    voice: selectedVoice,
                    systemPromptTokens: promptTokens,
                    maxSteps: maxSteps,
                    verbose: verbose)
                responseSamples = result.audio
                textTokens = result.textTokens
            }

            let elapsed = CFAbsoluteTimeGetCurrent() - startTime
            let responseDuration = Double(responseSamples.count) / 24000.0
            let rtf = responseDuration > 0 ? elapsed / responseDuration : 0

            let outputURL = URL(fileURLWithPath: output)
            try WAVWriter.write(samples: responseSamples, sampleRate: 24000, to: outputURL)

            // Decode transcript
            var decodedTranscript: String?
            if let dec = spmDecoder, !textTokens.isEmpty {
                decodedTranscript = dec.decode(textTokens)
            }

            if json {
                var result: [String: Any] = [
                    "audio": output,
                    "voice": selectedVoice.rawValue,
                    "system_prompt": promptLabel,
                    "input_duration": round(duration * 100) / 100,
                    "output_duration": round(responseDuration * 100) / 100,
                    "elapsed": round(elapsed * 100) / 100,
                    "rtf": round(rtf * 100) / 100,
                    "text_tokens": textTokens.count
                ]
                if let t = decodedTranscript { result["transcript"] = t }
                let jsonData = try JSONSerialization.data(
                    withJSONObject: result, options: [.prettyPrinted, .sortedKeys])
                print(String(data: jsonData, encoding: .utf8) ?? "{}")
            } else {
                print("Response: \(String(format: "%.2f", responseDuration))s (\(textTokens.count) text tokens)")
                print("Time: \(String(format: "%.2f", elapsed))s")
                if responseDuration > 0 {
                    print("RTF: \(String(format: "%.2f", rtf))")
                }
                if let t = decodedTranscript, transcript {
                    print("Transcript: \(t)")
                }
                print("Saved to \(output)")
            }
        }
    }
}
