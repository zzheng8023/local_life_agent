import Foundation
import Vision
import AppKit

// OCR Tool using macOS Vision framework
// Usage: ./ocr_tool <image_path>
// Output: recognized text to stdout, errors to stderr

func recognizeText(from imagePath: String) {
    guard let image = NSImage(contentsOfFile: imagePath),
          let cgImage = image.cgImage(forProposedRect: nil, context: nil, hints: nil) else {
        fputs("Error: Cannot load image at \(imagePath)\n", stderr)
        exit(1)
    }

    let request = VNRecognizeTextRequest { request, error in
        if let error = error {
            fputs("Error: \(error.localizedDescription)\n", stderr)
            exit(2)
        }

        guard let observations = request.results as? [VNRecognizedTextObservation] else {
            fputs("Error: No text recognized\n", stderr)
            exit(0)
        }

        var allText: [String] = []
        for observation in observations {
            if let topCandidate = observation.topCandidates(1).first {
                allText.append(topCandidate.string)
            }
        }

        if allText.isEmpty {
            exit(0)
        }

        print(allText.joined(separator: "\n"))
        exit(0)
    }

    // Configure for best recognition
    request.recognitionLevel = .accurate
    request.usesLanguageCorrection = true
    // Auto-detect language (supports zh, en, ja, ko, etc.)
    request.recognitionLanguages = ["zh-Hans", "zh-Hant", "en-US", "ja-JP", "ko-KR"]

    let handler = VNImageRequestHandler(cgImage: cgImage, options: [:])

    DispatchQueue.global().async {
        do {
            try handler.perform([request])
        } catch {
            fputs("Error: \(error.localizedDescription)\n", stderr)
            exit(3)
        }
    }

    // Keep main thread alive
    RunLoop.main.run(until: Date(timeIntervalSinceNow: 60))
}

func main() {
    let args = CommandLine.arguments
    guard args.count >= 2 else {
        fputs("Usage: ocr_tool <image_path>\n", stderr)
        exit(1)
    }

    let imagePath = args[1]
    recognizeText(from: imagePath)
}

main()
