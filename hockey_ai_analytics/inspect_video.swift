import Foundation
import AVFoundation
import AppKit

let path = CommandLine.arguments[1]
let out = CommandLine.arguments[2]
let asset = AVURLAsset(url: URL(fileURLWithPath: path))
let generator = AVAssetImageGenerator(asset: asset)
generator.appliesPreferredTrackTransform = true
generator.requestedTimeToleranceBefore = .zero
generator.requestedTimeToleranceAfter = .zero
print("duration_seconds=\(CMTimeGetSeconds(asset.duration))")
for track in asset.tracks(withMediaType: .video) {
    print("natural_size=\(track.naturalSize) nominal_fps=\(track.nominalFrameRate)")
}
for second in 0...14 {
    let t = CMTime(seconds: Double(second), preferredTimescale: 600)
    var actual = CMTime.zero
    do {
        let cg = try generator.copyCGImage(at: t, actualTime: &actual)
        let rep = NSBitmapImageRep(cgImage: cg)
        let data = rep.representation(using: .png, properties: [:])!
        try data.write(to: URL(fileURLWithPath: "\(out)/v3_\(second).png"))
        print("sample=\(second) actual_pts=\(CMTimeGetSeconds(actual))")
    } catch { print("sample=\(second) error=\(error)") }
}
