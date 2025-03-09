package main

import (
	"compress/gzip"
	"encoding/binary"
	"fmt"
	"io"
	"log"
	"math"
	"math/rand/v2"
	"net/http"
	"os"
	"path/filepath"

	"paragon" // Replace with your actual Paragon package import path
)

const (
	baseURL  = "https://storage.googleapis.com/cvdf-datasets/mnist/"
	mnistDir = "mnist_data"
)

func main() {
	fmt.Println("V5-IMPLEMENTATION-13")

	if err := ensureMNISTDownloads(mnistDir); err != nil {
		log.Fatalf("Failed to download MNIST data: %v", err)
	}

	inputs, targets, err := loadMNISTTrainingData(mnistDir)
	if err != nil {
		log.Fatalf("Failed to load MNIST data: %v", err)
	}
	fmt.Printf("Loaded %d training samples\n", len(inputs))

	// Initial network with Leaky ReLU (MLP for MNIST)
	layerSizes := []struct{ Width, Height int }{
		{28, 28}, // Input
		{16, 16}, // Hidden 1
		{10, 1},  // Output
	}
	activations := []string{"leaky_relu", "leaky_relu", "softmax"}
	fullyConnected := []bool{true, false, true}

	nn := paragon.NewNetwork(layerSizes, activations, fullyConnected)
	// nn.Debug = true

	// Training parameters
	epochsPerPhase := 50
	learningRate := 0.01
	plateauThreshold := 0.001
	plateauLimit := 3
	plateauCount := 0
	hasAddedNeurons := false
	targetLayerForNeurons := 1
	totalEpochs := 0
	maxTotalEpochs := 5

	// Training loop with dynamic growth
	for totalEpochs < maxTotalEpochs {
		prevLoss := math.Inf(1)
		phaseEpochs := 0
		for epoch := 0; epoch < epochsPerPhase && totalEpochs < maxTotalEpochs; epoch++ {
			totalLoss := 0.0
			perm := rand.Perm(len(inputs))
			shuffledInputs := make([][][]float64, len(inputs))
			shuffledTargets := make([][][]float64, len(targets))
			for i, p := range perm {
				shuffledInputs[i] = inputs[p]
				shuffledTargets[i] = targets[p]
			}
			for b := 0; b < len(shuffledInputs); b++ {
				nn.Forward(shuffledInputs[b]) // Use standard Forward for MNIST
				loss := nn.ComputeLoss(shuffledTargets[b])
				if math.IsNaN(loss) {
					fmt.Printf("NaN loss detected at sample %d, epoch %d\n", b, totalEpochs)
					continue
				}
				totalLoss += loss
				nn.Backward(shuffledTargets[b], learningRate)
			}
			avgLoss := totalLoss / float64(len(inputs))
			fmt.Printf("Epoch %d, Loss: %.4f\n", totalEpochs, avgLoss)

			// Check for plateau
			lossChange := math.Abs(prevLoss - avgLoss)
			if lossChange < plateauThreshold {
				plateauCount++
				fmt.Printf("Plateau detected (%d/%d), loss change: %.6f\n", plateauCount, plateauLimit, lossChange)
			} else {
				plateauCount = 0
			}
			prevLoss = avgLoss

			// Handle plateau actions
			if plateauCount >= plateauLimit {
				if !hasAddedNeurons {
					fmt.Println("Loss plateaued 3 times, adding 20 neurons to layer", targetLayerForNeurons)
					nn.AddNeuronsToLayer(targetLayerForNeurons, 20)
					hasAddedNeurons = true
					plateauCount = 0
				} else {
					fmt.Println("Loss plateaued again 3 times after adding neurons, adding a new layer")
					nn.AddLayer(2, 8, 8, "leaky_relu", true)
					targetLayerForNeurons = 2
					fmt.Println("Now adding neurons to new layer", targetLayerForNeurons)
					nn.AddNeuronsToLayer(targetLayerForNeurons, 20)
					plateauCount = 0
					phaseEpochs = epoch + 1
					break
				}
			}
			totalEpochs++
			phaseEpochs++
		}

		// Print output and accuracy after each phase
		nn.Forward(inputs[0])
		fmt.Printf("Output layer values after %d epochs:\n", totalEpochs)
		for y := 0; y < nn.Layers[nn.OutputLayer].Height; y++ {
			for x := 0; x < nn.Layers[nn.OutputLayer].Width; x++ {
				fmt.Printf("%.4f ", nn.Layers[nn.OutputLayer].Neurons[y][x].Value)
			}
			fmt.Println()
		}

		accuracy := computeAccuracy(nn, inputs, targets)
		fmt.Printf("Training accuracy on all samples after %d epochs: %.2f%%\n", totalEpochs, accuracy*100)

		if accuracy > 0.95 {
			break
		}
	}

	fmt.Println("Final network structure:")
	for i, layer := range nn.Layers {
		fmt.Printf("Layer %d: %dx%d\n", i, layer.Width, layer.Height)
	}
}

// computeAccuracy calculates the network's accuracy on a subset of data
func computeAccuracy(nn *paragon.Network, inputs [][][]float64, targets [][][]float64) float64 {
	correct := 0
	for i := range inputs {
		nn.Forward(inputs[i])
		// Extract output values into a []float64
		outputValues := make([]float64, nn.Layers[nn.OutputLayer].Width)
		for x := 0; x < nn.Layers[nn.OutputLayer].Width; x++ {
			outputValues[x] = nn.Layers[nn.OutputLayer].Neurons[0][x].Value
		}
		pred := argMax(outputValues)
		label := argMax(targets[i][0])
		if pred == label {
			correct++
		}
	}
	return float64(correct) / float64(len(inputs))
}

// argMax finds the index of the maximum value in a slice
func argMax(arr []float64) int {
	maxIdx := 0
	for i := 1; i < len(arr); i++ {
		if arr[i] > arr[maxIdx] {
			maxIdx = i
		}
	}
	return maxIdx
}

// ensureMNISTDownloads downloads and unzips MNIST files if they are not present
func ensureMNISTDownloads(targetDir string) error {
	if err := os.MkdirAll(targetDir, os.ModePerm); err != nil {
		return fmt.Errorf("failed to create directory %s: %w", targetDir, err)
	}

	files := []struct {
		compressed   string
		uncompressed string
	}{
		{"train-images-idx3-ubyte.gz", "train-images-idx3-ubyte"},
		{"train-labels-idx1-ubyte.gz", "train-labels-idx1-ubyte"},
	}

	for _, f := range files {
		compressedPath := filepath.Join(targetDir, f.compressed)
		uncompressedPath := filepath.Join(targetDir, f.uncompressed)

		if _, err := os.Stat(uncompressedPath); os.IsNotExist(err) {
			if _, err := os.Stat(compressedPath); os.IsNotExist(err) {
				fmt.Printf("Downloading %s...\n", f.compressed)
				if err := downloadFile(baseURL+f.compressed, compressedPath); err != nil {
					return fmt.Errorf("failed to download %s: %w", f.compressed, err)
				}
			}
			fmt.Printf("Unzipping %s...\n", f.compressed)
			if err := unzipFile(compressedPath, uncompressedPath); err != nil {
				return fmt.Errorf("failed to unzip %s: %w", f.compressed, err)
			}
			if err := os.Remove(compressedPath); err != nil {
				log.Printf("Warning: failed to remove %s: %v", compressedPath, err)
			}
		}
	}
	return nil
}

// downloadFile downloads a file from a URL to the specified path
func downloadFile(url, filepath string) error {
	resp, err := http.Get(url)
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	out, err := os.Create(filepath)
	if err != nil {
		return err
	}
	defer out.Close()

	_, err = io.Copy(out, resp.Body)
	return err
}

// unzipFile decompresses a .gz file to the specified output path
func unzipFile(src, dest string) error {
	fSrc, err := os.Open(src)
	if err != nil {
		return err
	}
	defer fSrc.Close()

	gzReader, err := gzip.NewReader(fSrc)
	if err != nil {
		return err
	}
	defer gzReader.Close()

	fDest, err := os.Create(dest)
	if err != nil {
		return err
	}
	defer fDest.Close()

	_, err = io.Copy(fDest, gzReader)
	return err
}

// loadMNISTTrainingData loads MNIST training images and labels into Paragon format
func loadMNISTTrainingData(dir string) (inputs [][][]float64, targets [][][]float64, err error) {
	imgPath := filepath.Join(dir, "train-images-idx3-ubyte")
	fImg, err := os.Open(imgPath)
	if err != nil {
		return nil, nil, fmt.Errorf("failed to open image file: %w", err)
	}
	defer fImg.Close()

	var imgHeader [16]byte
	if _, err := fImg.Read(imgHeader[:]); err != nil {
		return nil, nil, fmt.Errorf("failed to read image header: %w", err)
	}
	if magic := binary.BigEndian.Uint32(imgHeader[0:4]); magic != 2051 {
		return nil, nil, fmt.Errorf("invalid image magic number: %d", magic)
	}
	numImages := int(binary.BigEndian.Uint32(imgHeader[4:8]))
	rows := int(binary.BigEndian.Uint32(imgHeader[8:12]))
	cols := int(binary.BigEndian.Uint32(imgHeader[12:16]))
	if rows != 28 || cols != 28 {
		return nil, nil, fmt.Errorf("unexpected image dimensions: %dx%d", rows, cols)
	}

	lblPath := filepath.Join(dir, "train-labels-idx1-ubyte")
	fLbl, err := os.Open(lblPath)
	if err != nil {
		return nil, nil, fmt.Errorf("failed to open label file: %w", err)
	}
	defer fLbl.Close()

	var lblHeader [8]byte
	if _, err := fLbl.Read(lblHeader[:]); err != nil {
		return nil, nil, fmt.Errorf("failed to read label header: %w", err)
	}
	if magic := binary.BigEndian.Uint32(lblHeader[0:4]); magic != 2049 {
		return nil, nil, fmt.Errorf("invalid label magic number: %d", magic)
	}
	numLabels := int(binary.BigEndian.Uint32(lblHeader[4:8]))

	if numImages != numLabels {
		return nil, nil, fmt.Errorf("mismatch between number of images (%d) and labels (%d)", numImages, numLabels)
	}

	inputs = make([][][]float64, numImages)
	targets = make([][][]float64, numImages)
	imgBuf := make([]byte, 784) // 28x28 pixels

	for i := 0; i < numImages; i++ {
		if _, err := fImg.Read(imgBuf); err != nil {
			return nil, nil, fmt.Errorf("failed to read image %d: %w", i, err)
		}
		img := make([][]float64, 28)
		for r := 0; r < 28; r++ {
			img[r] = make([]float64, 28)
			for c := 0; c < 28; c++ {
				img[r][c] = float64(imgBuf[r*28+c]) / 255.0
			}
		}
		inputs[i] = img

		var lblByte [1]byte
		if _, err := fLbl.Read(lblByte[:]); err != nil {
			return nil, nil, fmt.Errorf("failed to read label %d: %w", i, err)
		}
		label := int(lblByte[0])
		targets[i] = labelToTarget(label)
	}

	return inputs, targets, nil
}

// labelToTarget converts an integer label to a one-hot encoded 1x10 slice
func labelToTarget(label int) [][]float64 {
	target := make([][]float64, 1)
	target[0] = make([]float64, 10)
	if label >= 0 && label < 10 {
		target[0][label] = 1.0
	}
	return target
}
