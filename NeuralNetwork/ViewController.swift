//
//  ViewController.swift
//  NeuralNetwork
//
//  Created by 保坂篤志 on 2024/06/30.
//

import UIKit
import Vision
import CoreML
import VideoToolbox

class ViewController: UIViewController, UIImagePickerControllerDelegate, UINavigationControllerDelegate {

    @IBOutlet weak var imageView: UIImageView!
    @IBOutlet weak var resultLabel: UILabel!

    let imagePicker = UIImagePickerController()

    override func viewDidLoad() {
        super.viewDidLoad()
        imagePicker.delegate = self
    }

    @IBAction func cameraButtonTapped(_ sender: UIButton) {
        imagePicker.sourceType = .camera
        present(imagePicker, animated: true, completion: nil)
    }

    @IBAction func albumButtonTapped(_ sender: UIButton) {
        imagePicker.sourceType = .photoLibrary
        present(imagePicker, animated: true, completion: nil)
    }

    func imagePickerController(_ picker: UIImagePickerController, didFinishPickingMediaWithInfo info: [UIImagePickerController.InfoKey : Any]) {
        if let pickedImage = info[UIImagePickerController.InfoKey.originalImage] as? UIImage {
            imageView.image = pickedImage
        }
        dismiss(animated: true, completion: nil)
    }

    @IBAction func classifyButtonTapped(_ sender: UIButton) {
        guard let image = imageView.image else {
            resultLabel.text = "画像を選択してください"
            return
        }

        classifyImage(image)
    }

    func classifyImage(_ image: UIImage) {
        guard let model = try? AI_human_classify_model(configuration: MLModelConfiguration()) else {
            print("モデルの読み込みに失敗しました")
            DispatchQueue.main.async {
                self.resultLabel.text = "モデルの読み込みに失敗しました"
            }
            return
        }
        
        let targetSize = CGSize(width: 224, height: 224)  // モデルが期待するサイズ
        guard let pixelBuffer = image.toCVPixelBuffer(targetSize: targetSize) else {
            print("画像の変換に失敗しました")
            return
        }
        
        do {
            let input = AI_human_classify_modelInput(image: pixelBuffer)
            let output = try model.prediction(input: input)
            
            dump(output.linear_0)
            dump(output.linear_0ShapedArray)
            
            let classificationResult = interpretOutput(output.linear_0)
            
            DispatchQueue.main.async {
                self.resultLabel.text = "結果: \(classificationResult.label) (\(Int(classificationResult.confidence * 100))%)"
            }
        } catch {
            print("予測エラー: \(error.localizedDescription)")
            DispatchQueue.main.async {
                self.resultLabel.text = "予測エラー: \(error.localizedDescription)"
            }
        }
    }

    func interpretOutput(_ output: MLMultiArray) -> (label: String, confidence: Float) {
        // モデルの出力を解釈してラベルと信頼度を返す
        // この部分はモデルの具体的な出力形式に応じて調整が必要です
        let probabilities = (0..<output.count).map { Float(truncating: output[$0]) }
        let maxIndex = probabilities.indices.max(by: { probabilities[$0] < probabilities[$1] }) ?? 0
        let maxProbability = probabilities[maxIndex]

        let labels = ["AI", "Human"] // モデルの出力に対応するラベルを設定
        return (labels[maxIndex], maxProbability)
    }
}

extension UIImage {
    func toCVPixelBuffer(targetSize: CGSize) -> CVPixelBuffer? {
        guard let resizedImage = self.resized(to: targetSize) else {
            return nil
        }

        let width = Int(targetSize.width)
        let height = Int(targetSize.height)

        // 以下は既存のコードと同じ
        let attributes: [String: Any] = [
            kCVPixelBufferCGImageCompatibilityKey as String: kCFBooleanTrue!,
            kCVPixelBufferCGBitmapContextCompatibilityKey as String: kCFBooleanTrue!
        ]

        var pixelBuffer: CVPixelBuffer?
        let status = CVPixelBufferCreate(kCFAllocatorDefault,
                                         width,
                                         height,
                                         kCVPixelFormatType_32ARGB,
                                         attributes as CFDictionary,
                                         &pixelBuffer)

        guard status == kCVReturnSuccess, let unwrappedPixelBuffer = pixelBuffer else {
            return nil
        }

        CVPixelBufferLockBaseAddress(unwrappedPixelBuffer, CVPixelBufferLockFlags(rawValue: 0))
        let pixelData = CVPixelBufferGetBaseAddress(unwrappedPixelBuffer)

        let rgbColorSpace = CGColorSpaceCreateDeviceRGB()
        guard let context = CGContext(data: pixelData,
                                      width: width,
                                      height: height,
                                      bitsPerComponent: 8,
                                      bytesPerRow: CVPixelBufferGetBytesPerRow(unwrappedPixelBuffer),
                                      space: rgbColorSpace,
                                      bitmapInfo: CGImageAlphaInfo.noneSkipFirst.rawValue) else {
            return nil
        }

        context.translateBy(x: 0, y: CGFloat(height))
        context.scaleBy(x: 1.0, y: -1.0)

        UIGraphicsPushContext(context)
        resizedImage.draw(in: CGRect(x: 0, y: 0, width: width, height: height))
        UIGraphicsPopContext()

        CVPixelBufferUnlockBaseAddress(unwrappedPixelBuffer, CVPixelBufferLockFlags(rawValue: 0))

        return unwrappedPixelBuffer
    }

    func resized(to size: CGSize) -> UIImage? {
        UIGraphicsBeginImageContextWithOptions(size, false, 0.0)
        defer { UIGraphicsEndImageContext() }
        draw(in: CGRect(origin: .zero, size: size))
        return UIGraphicsGetImageFromCurrentImageContext()
    }
}
