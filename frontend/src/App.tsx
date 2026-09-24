import { useState } from 'react'
import './App.css'

type PredictionResult = {
  prediction: number
  class_name: string
  confidence: number
}

function App() {
  const [selectedFile, setSelectedFile] = useState<File>()
  const [preview, setPreview] = useState<string>()
  const [result, setResult] = useState<PredictionResult>()
  const [loading, setLoading] = useState(false)

  function handleFileChange(e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0]

    if (file) {
      setSelectedFile(file)
      setPreview(URL.createObjectURL(file))
      setResult(undefined)
    }
  }

  async function handleSubmit() {
    if (!selectedFile) return

    try {
      setLoading(true)

      const formdata = new FormData()
      formdata.append("file", selectedFile)

      const API_PATH = "http://127.0.0.1:8000"

      const response = await fetch(`${API_PATH}/predict`, {
        method: "POST",
        body: formdata
      })

      if (!response.ok) {
        throw new Error("Prediction request failed")
      }

      const data: PredictionResult = await response.json()

      setResult(data)
    } catch (error) {
      console.error(error)
    } finally {
      setLoading(false)
    }
  }

  return (
    <main className="app">
      <div className="container">

        <header className="header">
          <h1>CIFAR-10 Classifier</h1>
          <p>
            Upload an image and let the CNN classify it.
          </p>
        </header>

        <section className="card">

          <div className="upload-area">

            {preview ? (
              <img
                src={preview}
                alt="Selected"
                className="preview"
              />
            ) : (
              <div className="placeholder">
                <span>📷</span>
                <p>Select an image to preview it</p>
              </div>
            )}

          </div>

          <label className="file-button">
            Choose Image
            <input
              type="file"
              accept="image/*"
              onChange={handleFileChange}
            />
          </label>

          <button
            className="predict-button"
            onClick={handleSubmit}
            disabled={!selectedFile || loading}
          >
            {loading ? "Predicting..." : "Predict"}
          </button>

          {result && (
            <div className="result">

              <p className="result-label">
                Prediction
              </p>

              <h2>
                {result.class_name}
              </h2>

              <p className="confidence">
                Confidence:{" "}
                <strong>
                  {result.confidence.toFixed(2)}%
                </strong>
              </p>

            </div>
          )}

        </section>

      </div>
    </main>
  )
}

export default App