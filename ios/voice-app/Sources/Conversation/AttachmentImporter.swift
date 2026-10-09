import Foundation
import PDFKit
import UIKit
import UniformTypeIdentifiers
import Vision

/// Turns a photo or a document into an `Attachment` the text-only model
/// can read: on-device text recognition for photos (Vision), text
/// extraction for PDFs (PDFKit) and plain-text formats. Nothing leaves the
/// phone.
enum AttachmentImporter {

    enum Failure: LocalizedError {
        case unreadablePhoto
        case noTextInPhoto
        case unreadableFile(String)
        case lockedPDF(String)
        case notUTF8(String)
        case unsupportedType(String)
        case noText(String)

        var errorDescription: String? {
            switch self {
            case .unreadablePhoto:
                return "Couldn't open that photo"
            case .noTextInPhoto:
                return "No text found in that photo"
            case .unreadableFile(let name):
                return "Couldn't open \(name)"
            case .lockedPDF(let name):
                return "\(name) is password-protected"
            case .notUTF8(let name):
                return "\(name) isn't UTF-8 text"
            case .unsupportedType(let type):
                return "Pie can't read \(type) files. Try a PDF or a text file."
            case .noText(let name):
                return "No text found in \(name)"
            }
        }
    }

    /// The most text kept from one attachment. The model is shown far less
    /// (`PieRuntimeConfig.attachmentTokensEach`); this bounds what is
    /// stored in the conversation file, which is rewritten after every
    /// reply, and how much of a long document is read at all.
    static let storedCharacterLimit = 20_000

    /// Longest side of a photo's thumbnail, in pixels.
    static let thumbnailSide: CGFloat = 256

    /// Common text formats the system does not type as text (source files
    /// without a registered UTType on iOS, mostly).
    private static let textExtensions: Set<String> = [
        "txt", "text", "md", "markdown", "json", "jsonl", "csv", "tsv", "log",
        "swift", "py", "rs", "js", "mjs", "ts", "tsx", "jsx", "java", "kt", "kts",
        "c", "h", "cc", "cpp", "hpp", "m", "mm", "go", "rb", "php", "cs", "dart",
        "scala", "lua", "pl", "r", "sql", "sh", "bash", "zsh", "fish",
        "yaml", "yml", "toml", "ini", "cfg", "conf", "xml", "html", "htm", "css",
        "tex", "bib", "srt", "vtt", "ipynb",
    ]

    // MARK: - Photos

    static func photo(_ imageData: Data) async throws -> Attachment {
        try await inBackground {
            guard let image = UIImage(data: imageData), let cgImage = image.cgImage else {
                throw Failure.unreadablePhoto
            }
            let text = try recognizeText(in: cgImage, orientation: CGImagePropertyOrientation(image.imageOrientation))
            guard !text.isEmpty else { throw Failure.noTextInPhoto }
            return Attachment(
                kind: .photo,
                name: "Photo",
                extractedText: String(text.prefix(storedCharacterLimit)),
                thumbnailJPEG: thumbnail(of: image)
            )
        }
    }

    private static func recognizeText(in image: CGImage, orientation: CGImagePropertyOrientation) throws -> String {
        let request = VNRecognizeTextRequest()
        request.recognitionLevel = .accurate
        request.usesLanguageCorrection = true
        request.automaticallyDetectsLanguage = true
        try VNImageRequestHandler(cgImage: image, orientation: orientation, options: [:]).perform([request])
        return (request.results ?? [])
            .compactMap { $0.topCandidates(1).first?.string }
            .joined(separator: "\n")
            .trimmingCharacters(in: .whitespacesAndNewlines)
    }

    private static func thumbnail(of image: UIImage) -> Data? {
        let size = image.size
        guard size.width > 0, size.height > 0 else { return nil }
        let scale = min(1, thumbnailSide / max(size.width, size.height))
        let target = CGSize(width: (size.width * scale).rounded(), height: (size.height * scale).rounded())
        let format = UIGraphicsImageRendererFormat()
        format.scale = 1
        format.opaque = true
        return UIGraphicsImageRenderer(size: target, format: format).jpegData(withCompressionQuality: 0.7) { _ in
            image.draw(in: CGRect(origin: .zero, size: target))
        }
    }

    // MARK: - Files

    static func file(at url: URL) async throws -> Attachment {
        try await inBackground {
            // Files from the document picker live outside the sandbox and
            // are readable only inside this bracket.
            let scoped = url.startAccessingSecurityScopedResource()
            defer { if scoped { url.stopAccessingSecurityScopedResource() } }

            let name = url.lastPathComponent
            let type = (try? url.resourceValues(forKeys: [.contentTypeKey]).contentType)
                ?? UTType(filenameExtension: url.pathExtension)

            let text: String
            if type?.conforms(to: .pdf) == true {
                text = try pdfText(at: url, name: name)
            } else if type?.conforms(to: .rtf) == true {
                text = try richText(at: url, name: name)
            } else if type?.conforms(to: .text) == true || textExtensions.contains(url.pathExtension.lowercased()) {
                text = try utf8Text(at: url, name: name)
            } else {
                let described = type?.localizedDescription
                    ?? (url.pathExtension.isEmpty ? name : url.pathExtension.uppercased())
                throw Failure.unsupportedType(described)
            }

            let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !trimmed.isEmpty else { throw Failure.noText(name) }
            return Attachment(kind: .file, name: name, extractedText: String(trimmed.prefix(storedCharacterLimit)))
        }
    }

    private static func pdfText(at url: URL, name: String) throws -> String {
        guard let document = PDFDocument(url: url) else { throw Failure.unreadableFile(name) }
        guard !document.isLocked else { throw Failure.lockedPDF(name) }
        var pages: [String] = []
        var length = 0
        // A long PDF is read only as far as will be kept.
        for index in 0..<document.pageCount where length < storedCharacterLimit {
            guard let text = document.page(at: index)?.string, !text.isEmpty else { continue }
            pages.append(text)
            length += text.count
        }
        return pages.joined(separator: "\n\n")
    }

    private static func richText(at url: URL, name: String) throws -> String {
        do {
            return try NSAttributedString(
                url: url,
                options: [.documentType: NSAttributedString.DocumentType.rtf],
                documentAttributes: nil
            ).string
        } catch {
            throw Failure.unreadableFile(name)
        }
    }

    private static func utf8Text(at url: URL, name: String) throws -> String {
        // Read only what can be kept: four bytes per character covers the
        // worst case of UTF-8.
        let data: Data
        do {
            let handle = try FileHandle(forReadingFrom: url)
            defer { try? handle.close() }
            data = try handle.read(upToCount: storedCharacterLimit * 4) ?? Data()
        } catch {
            throw Failure.unreadableFile(name)
        }
        // The cut may land inside a multi-byte character; up to three
        // trailing bytes can belong to it.
        for drop in 0...min(3, data.count) {
            if let text = String(data: data.dropLast(drop), encoding: .utf8) {
                return text
            }
        }
        throw Failure.notUTF8(name)
    }

    // MARK: - Threading

    /// Vision, PDFKit and file reads all block; run them off the main
    /// thread so the composer stays responsive while a document is read.
    private static func inBackground<T>(_ work: @escaping () throws -> T) async throws -> T {
        try await withCheckedThrowingContinuation { continuation in
            DispatchQueue.global(qos: .userInitiated).async {
                continuation.resume(with: Result { try work() })
            }
        }
    }
}

private extension CGImagePropertyOrientation {
    /// Vision reads the raw pixels of a `CGImage`; this says which way up
    /// the photo actually is.
    init(_ orientation: UIImage.Orientation) {
        switch orientation {
        case .up: self = .up
        case .upMirrored: self = .upMirrored
        case .down: self = .down
        case .downMirrored: self = .downMirrored
        case .left: self = .left
        case .leftMirrored: self = .leftMirrored
        case .right: self = .right
        case .rightMirrored: self = .rightMirrored
        @unknown default: self = .up
        }
    }
}
