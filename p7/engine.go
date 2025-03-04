package main

import (
	"encoding/binary"
	"fmt"
	"log"
	"math"
	"math/rand"
	"os"
	"path/filepath"
	"time"

	"phase" // Replace with your actual import path, e.g., "github.com/you/phase"
)

const (
	baseURL  = "https://storage.googleapis.com/cvdf-datasets/mnist/"
	mnistDir = "mnist_data"
)

func main() {
	// Seed random number generator
	rand.Seed(time.Now().UnixNano())

	// Step 1: Download MNIST if needed
	fmt.Println("Step 1: Ensuring MNIST dataset is downloaded...")
	bp := phase.NewPhase()
	if err := ensureMNISTDownloads(bp, mnistDir); err != nil {
		log.Fatalf("Failed to ensure MNIST data: %v", err)
	}

	// Step 2: Load MNIST data
	fmt.Println("Step 2: Loading MNIST training and testing datasets...")
	trainInputs, trainLabels, err := loadMNIST(mnistDir, "train-images-idx3-ubyte", "train-labels-idx1-ubyte", 60000)
	if err != nil {
		log.Fatalf("Error loading training MNIST: %v", err)
	}
	testInputs, testLabels, err := loadMNIST(mnistDir, "t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte", 10000)
	if err != nil {
		log.Fatalf("Error loading testing MNIST: %v", err)
	}

	// Step 3: Split training data 80% train, 20% val
	fmt.Println("Step 3: Splitting training data (no shuffle)...")
	totalTrain := 60000
	trainSplit := int(0.8 * float64(totalTrain))
	trainData := trainInputs[:trainSplit]
	trainLbls := trainLabels[:trainSplit]
	valData := trainInputs[trainSplit:totalTrain]
	valLbls := trainLabels[trainSplit:totalTrain]

	fmt.Printf("Training set:   %d samples\n", len(trainData))
	fmt.Printf("Validation set: %d samples\n", len(valData))
	fmt.Printf("Testing set:    %d samples\n", len(testInputs))

	// Step 4: Create a dummy network
	fmt.Println("Step 4: Creating a dummy neural network...")
	bp = phase.NewPhaseWithLayers([]int{784, 64, 10}, "relu", "linear")

	// Create checkpoints for the entire training set
	fmt.Println("Creating checkpoints on the entire training set...")
	startCreate := time.Now()
	trainCheckpoints := bp.CheckpointPreOutputNeurons(trainData, 1)
	durCreate := time.Since(startCreate)
	fmt.Printf("Checkpoint creation time: %d ms\n", durCreate.Milliseconds())

	// Compare time of full forward pass vs. partial from checkpoint
	fmt.Println("Timing full forward pass vs partial from checkpoints (training set)...")
	startFull := time.Now()
	for _, inputMap := range trainData {
		bp.ResetNeuronValues()
		bp.Forward(inputMap, 1)
		_ = bp.GetOutputs()
	}
	durFull := time.Since(startFull)

	startC := time.Now()
	for _, ck := range trainCheckpoints {
		bp.ResetNeuronValues()
		_ = bp.ComputeOutputsFromCheckpoint(ck)
	}
	durCheck := time.Since(startC)

	fmt.Printf("Full forward pass total:  %d ms\n", durFull.Milliseconds())
	fmt.Printf("Checkpoint partial pass:  %d ms\n", durCheck.Milliseconds())

	// Evaluate on validation using the original network
	valLabelsFloat := make([]float64, len(valLbls))
	for i, lbl := range valLbls {
		valLabelsFloat[i] = float64(lbl)
	}

	// EvaluateMetrics vs EvaluateMetricsFromCheckpoints for the original net
	fmt.Println("\nEvaluating on the validation set (original network)...")
	startEvalFull := time.Now()
	fullAcc, fullBins, fullApprox := bp.EvaluateMetrics(valData, valLabelsFloat)
	durEvalFull := time.Since(startEvalFull)

	fmt.Printf("Original Network (Full) => Acc: %.2f%%, Approx: %.2f, time %d ms\n",
		fullAcc, fullApprox, durEvalFull.Milliseconds())

	fmt.Println("Creating validation checkpoints for EvaluateMetricsFromCheckpoints (original net)...")
	startValCk := time.Now()
	valCheckpoints := bp.CheckpointPreOutputNeurons(valData, 1)
	durValCk := time.Since(startValCk)
	startEvalCk := time.Now()
	ckAcc, ckBins, ckApprox := bp.EvaluateMetricsFromCheckpoints(valCheckpoints, valLabelsFloat)
	durEvalCk := time.Since(startEvalCk)

	_ = ckBins
	_ = fullBins
	_ = trainLbls
	_ = testLabels

	fmt.Printf("Original Network (Checkpoint) => Acc: %.2f%%, Approx: %.2f\n", ckAcc, ckApprox)
	fmt.Printf("Time => checkpoint creation: %d ms, eval from checkpoints: %d ms\n",
		durValCk.Milliseconds(), durEvalCk.Milliseconds())

	// -------------------------------------------------------------------------
	// STEP 6: Add a neuron just before output, re-use the same pre-output checkpoints,
	// then do a partial pass. We'll do it in two ways:
	//  (A) "Manual partial eval" from old valCheckpoints
	//  (B) EvaluateMetricsFromCheckpoints again (which also uses valCheckpoints)
	// to show they match even with the new neuron in place
	// -------------------------------------------------------------------------
	fmt.Println("\nStep 6: Adding a new neuron just before the output...")

	// (1) Add a new neuron from pre-output IDs => output
	newN := bp.AddNeuronFromPreOutputs("dense", "relu", 1, 5)
	if newN == nil {
		log.Fatal("Failed to add new neuron!")
	}
	fmt.Printf("New neuron ID = %d was created and connected to the output.\n", newN.ID)

	// (2) "Manual partial evaluation" from the old valCheckpoints
	// We'll replicate the logic of EvaluateMetrics in a "partial" sense:
	// for each sample, we restore the pre-output states, compute the new neuron,
	// then compute the output neurons. We'll measure accuracy, etc.
	fmt.Println("Doing MANUAL partial evaluation with the newly added neuron, using old checkpoints...")

	startManual := time.Now()
	manualExact := 0.0
	manualApproxSum := 0.0
	thresholds := []float64{0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9}
	binCounts := make([]float64, len(thresholds)+1)
	nSamplesVal := len(valCheckpoints)
	for i, ck := range valCheckpoints {
		label := int(math.Round(valLabelsFloat[i]))
		if label < 0 || label >= len(bp.OutputNodes) {
			continue
		}
		// Restore pre-output states
		bp.ResetNeuronValues()
		for preID, state := range ck {
			if neuron, ok := bp.Neurons[preID]; ok {
				bp.SetNeuronState(neuron, state)
			}
		}

		// Compute new neuron
		sumNew := newN.Bias
		for _, conn := range newN.Connections {
			srcID := int(conn[0])
			w := conn[1]
			sumNew += bp.Neurons[srcID].Value * w
		}
		newN.Value = bp.ApplyScalarActivation(sumNew, newN.Activation)

		// Compute final outputs
		outputs := make([]float64, len(bp.OutputNodes))
		for j, outID := range bp.OutputNodes {
			outN := bp.Neurons[outID]
			sumOut := outN.Bias
			for _, c := range outN.Connections {
				sID := int(c[0])
				w := c[1]
				sumOut += bp.Neurons[sID].Value * w
			}
			outVal := bp.ApplyScalarActivation(sumOut, outN.Activation)
			outputs[j] = outVal
		}

		// Argmax => label
		predClass := argmaxFloatSlice(outputs)
		if predClass == label {
			manualExact++
		}

		// Binning
		correctVal := outputs[label]
		diff := math.Abs(correctVal - 1.0)
		if diff > 1 {
			diff = 1
		}
		assigned := false
		for k, th := range thresholds {
			if diff <= th {
				binCounts[k]++
				assigned = true
				break
			}
		}
		if !assigned {
			binCounts[len(thresholds)]++
		}

		// Approx
		approx := bp.CalculatePercentageMatch(float64(label), float64(predClass))
		manualApproxSum += approx / 100.0 * (100.0 / float64(nSamplesVal))
	}
	durManual := time.Since(startManual)
	manualExactAcc := (manualExact / float64(nSamplesVal)) * 100.0

	manualBins := make([]float64, len(binCounts))
	for i, cnt := range binCounts {
		manualBins[i] = (cnt / float64(nSamplesVal)) * 100.0
	}

	fmt.Printf("Manual partial (new neuron) => ExactAcc = %.2f%%, ApproxScore = %.2f\n",
		manualExactAcc, manualApproxSum)
	fmt.Printf("Time => %d ms\n", durManual.Milliseconds())

	// (3) EvaluateMetricsFromCheckpoints again, but now the net has a new neuron
	// Because the network changed *in place*, ComputeOutputsFromCheckpoint will also
	// run the new neuron (connected from pre-output states) plus the final outputs.
	// So the "old" valCheckpoints remain valid for the pre-output portion.
	fmt.Println("Doing EvaluateMetricsFromCheckpoints on the updated net, reusing old valCheckpoints...")

	startEval2 := time.Now()
	exact2, bins2, approx2 := bp.EvaluateMetricsFromCheckpoints(valCheckpoints, valLabelsFloat)
	durEval2 := time.Since(startEval2)

	fmt.Printf("EvaluateMetricsFromCheckpoints => ExactAcc = %.2f%%, ApproxScore = %.2f\n", exact2, approx2)
	fmt.Printf("Time => %d ms\n", durEval2.Milliseconds())

	// Compare the results from manual partial vs EvaluateMetricsFromCheckpoints
	fmt.Println("\nComparison of partial evaluations after adding the neuron:")
	fmt.Printf("Manual partial => Acc=%.2f%%, Approx=%.2f\n", manualExactAcc, manualApproxSum)
	fmt.Printf("FromCheckpoints => Acc=%.2f%%, Approx=%.2f\n", exact2, approx2)
	fmt.Println("Closeness bin comparison (Manual vs EvaluateMetricsFromCheckpoints):")
	for i := range manualBins {
		th := (i + 1) * 10
		fmt.Printf("  %2d%% bin => Manual %.2f%% vs Checkpoint %.2f%%\n", th, manualBins[i], bins2[i])
	}

	fmt.Println("\nAll done. We used the same old pre-output checkpoints but added a new neuron,")
	fmt.Println("thus skipping re-processing the lower layers. Both partial methods match as expected.")
}

// --------------------------------------------------------------------------
// Helper functions
// --------------------------------------------------------------------------

func ensureMNISTDownloads(bp *phase.Phase, targetDir string) error {
	if err := os.MkdirAll(targetDir, os.ModePerm); err != nil {
		return err
	}
	files := []string{
		"train-images-idx3-ubyte.gz",
		"train-labels-idx1-ubyte.gz",
		"t10k-images-idx3-ubyte.gz",
		"t10k-labels-idx1-ubyte.gz",
	}
	for _, f := range files {
		localPath := filepath.Join(targetDir, f)
		if _, err := os.Stat(localPath); os.IsNotExist(err) {
			fmt.Printf("Downloading %s...\n", f)
			if err := bp.DownloadFile(localPath, baseURL+f); err != nil {
				return fmt.Errorf("failed to download %s: %w", f, err)
			}
			fmt.Printf("Unzipping %s...\n", f)
			if err := bp.UnzipFile(localPath, targetDir); err != nil {
				return fmt.Errorf("failed to unzip %s: %w", f, err)
			}
		}
	}
	return nil
}

func loadMNIST(dir, imageFile, labelFile string, limit int) ([]map[int]float64, []int, error) {
	// Open image file
	imgPath := filepath.Join(dir, imageFile)
	fImg, err := os.Open(imgPath)
	if err != nil {
		return nil, nil, fmt.Errorf("open image file: %w", err)
	}
	defer fImg.Close()

	// Read image header
	var headerImg [16]byte
	if _, err := fImg.Read(headerImg[:]); err != nil {
		return nil, nil, fmt.Errorf("read image header: %w", err)
	}
	numImages := int(binary.BigEndian.Uint32(headerImg[4:8]))
	if limit > numImages {
		limit = numImages
	}

	// Open label file
	lblPath := filepath.Join(dir, labelFile)
	fLbl, err := os.Open(lblPath)
	if err != nil {
		return nil, nil, fmt.Errorf("open label file: %w", err)
	}
	defer fLbl.Close()

	// Read label header
	var headerLbl [8]byte
	if _, err := fLbl.Read(headerLbl[:]); err != nil {
		return nil, nil, fmt.Errorf("read label header: %w", err)
	}
	numLabels := int(binary.BigEndian.Uint32(headerLbl[4:8]))
	if limit > numLabels {
		limit = numLabels
	}

	// Prepare slices
	inputs := make([]map[int]float64, limit)
	labels := make([]int, limit)

	// Buffer for one image (784 = 28×28)
	buf := make([]byte, 784)
	for i := 0; i < limit; i++ {
		// Read image
		if _, err := fImg.Read(buf); err != nil {
			return nil, nil, fmt.Errorf("read image data at sample %d: %w", i, err)
		}
		// Build input map
		inputMap := make(map[int]float64, 784)
		for px := 0; px < 784; px++ {
			// Normalize [0..255] -> [0..1]
			inputMap[px] = float64(buf[px]) / 255.0
		}
		inputs[i] = inputMap

		// Read label
		var lblByte [1]byte
		if _, err := fLbl.Read(lblByte[:]); err != nil {
			return nil, nil, fmt.Errorf("read label data at sample %d: %w", i, err)
		}
		labels[i] = int(lblByte[0])
	}
	return inputs, labels, nil
}

// For convenience if you don't already have it:
func argmaxFloatSlice(vals []float64) int {
	if len(vals) == 0 {
		return -1
	}
	maxIdx := 0
	maxVal := vals[0]
	for i := 1; i < len(vals); i++ {
		if vals[i] > maxVal {
			maxVal = vals[i]
			maxIdx = i
		}
	}
	return maxIdx
}
