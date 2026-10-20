import Foundation
import Vision
import AppKit
let inputs = try JSONSerialization.jsonObject(with: Data(contentsOf: URL(fileURLWithPath: CommandLine.arguments[1]))) as! [[String:Any]]
var output = [[String:Any]]()
for item in inputs {
 let path = item["path"] as! String
 var row = item
 do {
  let request = VNRecognizeTextRequest()
  request.recognitionLevel = .accurate
  request.usesCPUOnly = true
  request.usesLanguageCorrection = false
  request.recognitionLanguages = ["en-US"]
  request.minimumTextHeight = 0.03
  let handler = VNImageRequestHandler(url: URL(fileURLWithPath: path), options: [:])
  try handler.perform([request])
  row["observations"] = (request.results ?? []).compactMap { observation -> [String:Any]? in
   guard let t = observation.topCandidates(1).first else {return nil}
   return ["text":t.string,"confidence":t.confidence]
  }
 } catch {row["error"] = String(describing:error as NSError)}
 output.append(row)
}
let data = try JSONSerialization.data(withJSONObject:output,options:[.prettyPrinted,.sortedKeys])
try data.write(to:URL(fileURLWithPath:CommandLine.arguments[2]))
