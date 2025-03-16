package main

import (
	"compress/gzip"
	"encoding/binary"
	"fmt"
	"io"
	"log"
	"math"
	"math/rand"
	"net/http"
	"os"
	"path/filepath"
	"time"

	"paragon" // Replace with your actual Paragon package import path
)

const (
	baseURL   = "https://storage.googleapis.com/cvdf-datasets/mnist/"
	mnistDir  = "mnist_data"
	modelDir  = "models"
	modelFile = "mnist_model.json"
)

func main() {
	rand.Seed(time.Now().UnixNano()) // seed for reproducibility

	fmt.Println("MNIST Example with On-Demand Training")

	// 1) Load MNIST data
	trainInputs, trainTargets, err := loadMNISTData(mnistDir, true)
	if err != nil {
		log.Fatalf("Failed to load training data: %v", err)
	}
	testInputs, testTargets, err := loadMNISTData(mnistDir, false)
	if err != nil {
		log.Fatalf("Failed to load test data: %v", err)
	}

	// 2) Split training data (80/20 split for train/val)
	trIn, trTarg, valIn, valTarg := paragon.SplitDataset(trainInputs, trainTargets, 0.8)
	fmt.Printf("Train samples: %d, Validation samples: %d, Test samples: %d\n",
		len(trIn), len(valIn), len(testInputs))

	// 3) Define the network architecture
	layerSizes := []struct{ Width, Height int }{
		{28, 28}, // Input layer
		{16, 16}, // Hidden layer
		{10, 1},  // Output layer
	}
	activations := []string{"leaky_relu", "leaky_relu", "softmax"}
	fullyConnected := []bool{true, false, true}

	modelPath := filepath.Join(modelDir, modelFile)

	var nn *paragon.Network

	// 4) Check if a trained model exists
	if _, err := os.Stat(modelPath); err == nil {
		// Model found; load it
		fmt.Println("✅ Found pre-trained model. Loading...")
		nn = paragon.NewNetwork(layerSizes, activations, fullyConnected)
		if err := nn.LoadFromJSON(modelPath); err != nil {
			log.Fatalf("Failed to load model from JSON: %v", err)
		}

		fmt.Println("✅ Model loaded. No retraining performed.")
	} else {
		// 5) Model not found; create a new network and train
		fmt.Println("No pre-trained model found; creating and training new network...")
		nn = paragon.NewNetwork(layerSizes, activations, fullyConnected)

		trainer := &paragon.Trainer{
			Network: nn,
			Config: paragon.TrainConfig{
				Epochs:           5,    // e.g. 5
				LearningRate:     0.01, // base LR
				PlateauThreshold: 0.001,
				PlateauLimit:     3,
				EarlyStopAcc:     0.95, // stop if val accuracy > 95%
				Debug:            true,
			},
		}

		fmt.Println("Starting training/validation loop...")
		trainer.TrainWithValidation(
			trIn, trTarg,
			valIn, valTarg,
			testInputs, testTargets,
		)

		// 6) Save the newly trained model
		if err := nn.SaveToJSON(modelPath); err != nil {
			log.Fatalf("Failed to save model to JSON: %v", err)
		}

		fmt.Println("✅ Model trained and saved.")
	}

	// 7) Evaluate final performance
	trainAcc := paragon.ComputeAccuracy(nn, trIn, trTarg)
	valAcc := paragon.ComputeAccuracy(nn, valIn, valTarg)
	testAcc := paragon.ComputeAccuracy(nn, testInputs, testTargets)

	fmt.Println("\nFinal model performance:")
	fmt.Printf("Training Accuracy: %.2f%%\n", trainAcc*100)
	fmt.Printf("Validation Accuracy: %.2f%%\n", valAcc*100)
	fmt.Printf("Test Accuracy: %.2f%%\n", testAcc*100)

	// 8) Evaluate with ADHD
	paragon.EvaluateWithADHD(nn, testInputs, testTargets)
	adhdPerf := nn.Performance
	fmt.Printf("\n🔍 ADHD Analysis Results:\n")
	fmt.Printf("Model Performance Score: %.2f\n", adhdPerf.Score)
	for bucket, data := range adhdPerf.Buckets {
		fmt.Printf("%s → %d samples\n", bucket, data.Count)
	}

	fmt.Println("\nAll done!")
}

func OLDmain() {
	fmt.Println("V5-IMPLEMENTATION-13 (80/20 Train/Test Split with Model Detection)")

	if err := ensureMNISTDownloads(mnistDir); err != nil {
		log.Fatalf("Failed to download MNIST data: %v", err)
	}

	// Load training and test data
	inputs, targets, err := loadMNISTData(mnistDir, true)
	if err != nil {
		log.Fatalf("Failed to load MNIST training data: %v", err)
	}
	testInputs, testTargets, err := loadMNISTData(mnistDir, false)
	if err != nil {
		log.Fatalf("Failed to load MNIST test data: %v", err)
	}

	// Split training data into 80% train, 20% validation
	trainSize := int(0.8 * float64(len(inputs)))
	perm := rand.Perm(len(inputs))
	trainInputs := make([][][]float64, trainSize)
	trainTargets := make([][][]float64, trainSize)
	valInputs := make([][][]float64, len(inputs)-trainSize)
	valTargets := make([][][]float64, len(inputs)-trainSize)

	for i, p := range perm {
		if i < trainSize {
			trainInputs[i] = inputs[p]
			trainTargets[i] = targets[p]
		} else {
			valInputs[i-trainSize] = inputs[p]
			valTargets[i-trainSize] = targets[p]
		}
	}

	fmt.Printf("Training samples: %d, Validation samples: %d, Test samples: %d\n",
		len(trainInputs), len(valInputs), len(testInputs))

	var nn *paragon.Network
	modelPath := filepath.Join(modelDir, modelFile)

	// Define the network architecture (must match the saved model)
	layerSizes := []struct{ Width, Height int }{
		{28, 28}, // Input layer
		{16, 16}, // Hidden layer
		{10, 1},  // Output layer
	}
	activations := []string{"leaky_relu", "leaky_relu", "softmax"}
	fullyConnected := []bool{true, false, true}

	// Check if model exists
	if _, err := os.Stat(modelPath); err == nil {
		fmt.Println("✅ Loading pre-trained model...")
		nn = paragon.NewNetwork(layerSizes, activations, fullyConnected) // Initialize nn
		err = nn.LoadFromBinary(modelPath)
		if err != nil {
			log.Fatalf("⚠️ Failed to load existing model: %v", err)
		}
		fmt.Println("✅ Using pre-trained model.")
	} else {
		fmt.Println("No pre-trained model found, starting training...")
		nn = paragon.NewNetwork(layerSizes, activations, fullyConnected)

		epochsPerPhase := 50
		learningRate := 0.01
		plateauThreshold := 0.001
		plateauLimit := 3
		plateauCount := 0
		hasAddedNeurons := false
		targetLayerForNeurons := 1
		totalEpochs := 0
		maxTotalEpochs := 5

		for totalEpochs < maxTotalEpochs {
			prevLoss := math.Inf(1)
			for epoch := 0; epoch < epochsPerPhase && totalEpochs < maxTotalEpochs; epoch++ {
				totalLoss := 0.0
				perm := rand.Perm(len(trainInputs))
				shuffledInputs := make([][][]float64, len(trainInputs))
				shuffledTargets := make([][][]float64, len(trainTargets))
				for i, p := range perm {
					shuffledInputs[i] = trainInputs[p]
					shuffledTargets[i] = trainTargets[p]
				}
				for b := 0; b < len(shuffledInputs); b++ {
					nn.Forward(shuffledInputs[b])
					loss := nn.ComputeLoss(shuffledTargets[b])
					if math.IsNaN(loss) {
						fmt.Printf("NaN loss detected at sample %d, epoch %d\n", b, totalEpochs)
						continue
					}
					totalLoss += loss
					nn.Backward(shuffledTargets[b], learningRate)
				}
				avgLoss := totalLoss / float64(len(trainInputs))
				fmt.Printf("Epoch %d, Training Loss: %.4f\n", totalEpochs, avgLoss)

				lossChange := math.Abs(prevLoss - avgLoss)
				if lossChange < plateauThreshold {
					plateauCount++
					fmt.Printf("Plateau detected (%d/%d), loss change: %.6f\n", plateauCount, plateauLimit, lossChange)
				} else {
					plateauCount = 0
				}
				prevLoss = avgLoss

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
						break
					}
				}
				totalEpochs++
			}

			trainAcc := computeAccuracy(nn, trainInputs, trainTargets)
			valAcc := computeAccuracy(nn, valInputs, valTargets)
			testAcc := computeAccuracy(nn, testInputs, testTargets)

			fmt.Printf("After %d epochs:\n", totalEpochs)
			fmt.Printf("Training Accuracy: %.2f%%\n", trainAcc*100)
			fmt.Printf("Validation Accuracy: %.2f%%\n", valAcc*100)
			fmt.Printf("Test Accuracy: %.2f%%\n", testAcc*100)

			if valAcc > 0.95 {
				fmt.Println("Reached 95% validation accuracy, stopping training")
				break
			}
		}

		// Save trained model
		if err := nn.SaveToBinary(modelPath); err != nil { // Use modelPath here
			log.Fatalf("⚠️ Failed to save trained model: %v", err)
		}
		fmt.Println("✅ Model trained and saved.")
	}

	trainAcc := computeAccuracy(nn, trainInputs, trainTargets)
	valAcc := computeAccuracy(nn, valInputs, valTargets)
	testAcc := computeAccuracy(nn, testInputs, testTargets)

	fmt.Println("\nModel performance:")
	fmt.Printf("Training Accuracy: %.2f%%\n", trainAcc*100)
	fmt.Printf("Validation Accuracy: %.2f%%\n", valAcc*100)
	fmt.Printf("Test Accuracy: %.2f%%\n", testAcc*100)

	// Run ADHD Analysis after evaluating test accuracy
	EvaluateModelWithADHD(nn, testInputs, testTargets, "adhd_results.json")

	//printRandomSamples(nn, trainInputs, trainTargets, "Training")
	//printRandomSamples(nn, testInputs, testTargets, "Test")

	fmt.Println("\nFinal network structure:")
	for i, layer := range nn.Layers {
		fmt.Printf("Layer %d: %dx%d\n", i, layer.Width, layer.Height)
	}
}

// computeAccuracy calculates the network's accuracy on a subset of data
func computeAccuracy(nn *paragon.Network, inputs [][][]float64, targets [][][]float64) float64 {
	correct := 0
	for i := range inputs {
		nn.Forward(inputs[i])
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
		{"t10k-images-idx3-ubyte.gz", "t10k-images-idx3-ubyte"},
		{"t10k-labels-idx1-ubyte.gz", "t10k-labels-idx1-ubyte"},
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

// loadMNISTData loads MNIST data (training or test) into Paragon format
func loadMNISTData(dir string, isTraining bool) (inputs [][][]float64, targets [][][]float64, err error) {
	prefix := "train"
	if !isTraining {
		prefix = "t10k"
	}
	imgPath := filepath.Join(dir, prefix+"-images-idx3-ubyte")
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

	lblPath := filepath.Join(dir, prefix+"-labels-idx1-ubyte")
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
	imgBuf := make([]byte, 784)

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

// argMaxNeurons finds the index of the neuron with the maximum value
func argMaxNeurons(neurons []*paragon.Neuron) int {
	maxIdx := 0
	maxVal := neurons[0].Value
	for i, neuron := range neurons {
		if neuron.Value > maxVal {
			maxVal = neuron.Value
			maxIdx = i
		}
	}
	return maxIdx
}

// printRandomSamples prints 10 random samples with digit labels
func printRandomSamples(nn *paragon.Network, inputs [][][]float64, targets [][][]float64, datasetName string) {
	fmt.Printf("\nRandom samples from %s set:\n", datasetName)
	indices := rand.Perm(len(inputs))
	sampleCount := 10
	if sampleCount > len(indices) {
		sampleCount = len(indices)
	}
	for i := 0; i < sampleCount; i++ {
		idx := indices[i]
		sampleInput := inputs[idx]
		sampleTarget := targets[idx]

		nn.Forward(sampleInput)
		outputNeurons := nn.Layers[nn.OutputLayer].Neurons[0]
		expected := argMax(sampleTarget[0])
		predicted := argMaxNeurons(outputNeurons)

		fmt.Printf("Sample %d: Expected: %d, Predicted: %d\n", idx, expected, predicted)
		fmt.Println("Neuron outputs:")
		for digit, neuron := range outputNeurons {
			fmt.Printf("  Digit %d: %.4f  ", digit, neuron.Value)
		}
		fmt.Println("\n---------------------------")
	}
}

func EvaluateModelWithADHD(nn *paragon.Network, inputs [][][]float64, targets [][][]float64, filename string) {
	// Use the same layer structure as the trained model
	layerSizes := []struct{ Width, Height int }{
		{28, 28},
		{16, 16},
		{10, 1},
	}
	activations := []string{"leaky_relu", "leaky_relu", "softmax"}
	fullyConnected := []bool{true, false, true}

	adhdNetwork := paragon.NewNetwork(layerSizes, activations, fullyConnected) // ✅ Correctly initialized

	expectedOutputs := make([]float64, len(inputs))
	actualOutputs := make([]float64, len(inputs))

	// Run forward pass on each sample
	for i := range inputs {
		nn.Forward(inputs[i]) // Get the model's output

		// Extract the predicted and expected values
		outputValues := make([]float64, nn.Layers[nn.OutputLayer].Width)
		for x := 0; x < nn.Layers[nn.OutputLayer].Width; x++ {
			outputValues[x] = nn.Layers[nn.OutputLayer].Neurons[0][x].Value
		}
		predicted := argMax(outputValues) // Get the model's predicted class
		expected := argMax(targets[i][0]) // Get the actual class label

		expectedOutputs[i] = float64(expected)
		actualOutputs[i] = float64(predicted)
	}

	// Run ADHD evaluation
	adhdNetwork.EvaluateModel(expectedOutputs, actualOutputs)

	// Print ADHD results
	fmt.Printf("\n🔍 ADHD Analysis Results:\n")
	fmt.Printf("Model Performance Score: %.2f\n", adhdNetwork.Performance.Score)
	for bucket, data := range adhdNetwork.Performance.Buckets {
		fmt.Printf("%s → %d samples\n", bucket, data.Count)
	}

	// Save results to file
	if err := adhdNetwork.SaveToJSON(filename); err != nil {
		fmt.Printf("⚠️ Failed to save ADHD results: %v\n", err)
	} else {
		fmt.Printf("✅ ADHD results saved to %s\n", filename)
	}
}
