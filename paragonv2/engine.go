package main

import (
	"encoding/binary"
	"fmt"
	"log"
	"math/rand"
	"os"
	"path/filepath"
	"runtime"
	"time"

	"phase" // Adjust this import path to your local Phase framework
)

const (
	baseURL          = "https://storage.googleapis.com/cvdf-datasets/mnist/"
	mnistDir         = "mnist_data"
	modelsDir        = "models"
	populationSize   = 10  // Number of networks in the population
	samplePercent    = 5   // Percentage of validation set to sample for quick mutation evaluation
	attemptsPerModel = 10  // Mutation attempts per network per generation
	totalGenerations = 100 // Total evolutionary generations
)

// getConcurrencyLimit returns 80% of available CPU cores (at least 1)
func getConcurrencyLimit() int {
	cores := runtime.NumCPU()
	limit := int(0.8 * float64(cores))
	if limit < 1 {
		limit = 1
	}
	return limit
}

func main() {
	rand.Seed(time.Now().UnixNano())

	// Ensure MNIST data is downloaded and unzipped.
	tempBP := phase.NewPhase() // temporary instance used for downloads/unzipping
	if err := ensureMNISTDownloads(tempBP, mnistDir); err != nil {
		log.Fatalf("Failed to download MNIST data: %v", err)
	}

	// Load MNIST training and test datasets.
	trainInputs, trainLabels, err := loadMNIST(mnistDir, "train-images-idx3-ubyte", "train-labels-idx1-ubyte", 60000)
	if err != nil {
		log.Fatalf("Error loading training data: %v", err)
	}
	testInputs, testLabels, err := loadMNIST(mnistDir, "t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte", 10000)
	if err != nil {
		log.Fatalf("Error loading test data: %v", err)
	}

	// Split training data into 80% training and 20% validation.
	split := int(0.8 * float64(len(trainInputs)))
	valInputs := trainInputs[split:]
	valLabels := trainLabels[split:]
	trainInputs = trainInputs[:split]
	trainLabels = trainLabels[:split]

	fmt.Printf("Training samples: %d\nValidation samples: %d\nTest samples: %d\n",
		len(trainInputs), len(valInputs), len(testInputs))

	// Convert labels to float64 for evaluation.
	valLabelsF := make([]float64, len(valLabels))
	for i, v := range valLabels {
		valLabelsF[i] = float64(v)
	}
	testLabelsF := make([]float64, len(testLabels))
	for i, v := range testLabels {
		testLabelsF[i] = float64(v)
	}

	// Initialize a population of networks.
	population := make([]*phase.Phase, populationSize)
	for i := 0; i < populationSize; i++ {
		population[i] = phase.NewPhaseWithLayers([]int{784, 64, 10}, "relu", "linear")
	}

	// Evaluate initial population on the full validation set.
	bestApproxGlobal := -1.0
	bestGlobalModel := population[0].Copy()
	for i, net := range population {
		_, _, approx := net.EvaluateMetrics(valInputs, valLabelsF)
		fmt.Printf("Initial population[%d] approx score: %.4f\n", i, approx)
		if approx > bestApproxGlobal {
			bestApproxGlobal = approx
			bestGlobalModel = net.Copy()
		}
	}
	fmt.Printf("Global best approx score: %.4f\n", bestApproxGlobal)

	concurrencyLimit := getConcurrencyLimit()
	fmt.Printf("Using concurrency limit = %d (80%% of CPU cores)\n", concurrencyLimit)

	// Evolutionary training loop.
	for gen := 1; gen <= totalGenerations; gen++ {
		fmt.Printf("\n=== Generation %d ===\n", gen)
		// Process each network in the population.
		for i, net := range population {
			// Select a small random sample from the validation set.
			sampleInputs, sampleLabels := getRandomSample(valInputs, valLabelsF, samplePercent)
			// Evaluate the current network on the sample using exact accuracy.
			sampleExact, _, _ := net.EvaluateMetrics(sampleInputs, sampleLabels)
			bestCandidate := net.Copy()
			bestCandidateExact := sampleExact

			// Run mutation attempts concurrently for this network.
			type mutationResult struct {
				net   *phase.Phase
				exact float64
			}
			resultsCh := make(chan mutationResult, attemptsPerModel)
			sem := make(chan struct{}, concurrencyLimit)

			for attempt := 0; attempt < attemptsPerModel; attempt++ {
				go func(attempt int) {
					sem <- struct{}{} // acquire a slot
					trial := net.Copy()
					mutateOne(trial)
					trialExact, _, _ := trial.EvaluateMetrics(sampleInputs, sampleLabels)
					resultsCh <- mutationResult{net: trial, exact: trialExact}
					<-sem // release the slot
				}(attempt)
			}

			// Collect mutation attempt results based on exact accuracy.
			for j := 0; j < attemptsPerModel; j++ {
				res := <-resultsCh
				if res.exact > bestCandidateExact {
					bestCandidate = res.net.Copy()
					bestCandidateExact = res.exact
				}
			}
			fmt.Printf("Population[%d]: Sample exact improved from %.4f to %.4f\n",
				i, sampleExact, bestCandidateExact)

			// Now, evaluate the best candidate on the full validation set using approx metric.
			_, _, fullApprox := bestCandidate.EvaluateMetrics(valInputs, valLabelsF)
			// For logging, we also fetch the original network's full approx.
			_, _, origApprox := net.EvaluateMetrics(valInputs, valLabelsF)
			fmt.Printf("Population[%d]: Full validation approx score: %.4f (baseline: %.4f)\n",
				i, fullApprox, origApprox)
			if fullApprox > origApprox {
				population[i] = bestCandidate.Copy()
			}
		} // End of per-population mutation attempts.

		// Evaluate the full population concurrently.
		type evalResult struct {
			index  int
			approx float64
		}
		evalCh := make(chan evalResult, populationSize)
		evalSem := make(chan struct{}, concurrencyLimit)
		for i, net := range population {
			go func(i int, net *phase.Phase) {
				evalSem <- struct{}{}
				_, _, approx := net.EvaluateMetrics(valInputs, valLabelsF)
				evalCh <- evalResult{index: i, approx: approx}
				<-evalSem
			}(i, net)
		}

		for i := 0; i < populationSize; i++ {
			res := <-evalCh
			fmt.Printf("Population[%d]: Full validation approx score: %.4f\n", res.index, res.approx)
			if res.approx > bestApproxGlobal {
				bestApproxGlobal = res.approx
				bestGlobalModel = population[res.index].Copy()
			}
		}
		fmt.Printf("Global best approx score so far: %.4f\n", bestApproxGlobal)
		// Save the current global best model.
		saveModel(bestGlobalModel, gen)
	}

	// Final evaluation on the test set.
	exactAcc, _, testApprox := bestGlobalModel.EvaluateMetrics(testInputs, testLabelsF)
	fmt.Printf("\n--- Final Best Network ---\n")
	fmt.Printf("Test Exact Accuracy: %.3f%%\n", exactAcc)
	fmt.Printf("Test Approx Score: %.4f\n", testApprox)
}

// ensureMNISTDownloads downloads and unzips the MNIST dataset files if needed.
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

// loadMNIST loads images and labels from MNIST files.
func loadMNIST(dir, imageFile, labelFile string, limit int) ([]map[int]float64, []int, error) {
	imgPath := filepath.Join(dir, imageFile)
	fImg, err := os.Open(imgPath)
	if err != nil {
		return nil, nil, err
	}
	defer fImg.Close()

	var headerImg [16]byte
	if _, err := fImg.Read(headerImg[:]); err != nil {
		return nil, nil, err
	}
	numImages := int(binary.BigEndian.Uint32(headerImg[4:8]))
	if limit > numImages {
		limit = numImages
	}

	lblPath := filepath.Join(dir, labelFile)
	fLbl, err := os.Open(lblPath)
	if err != nil {
		return nil, nil, err
	}
	defer fLbl.Close()

	var headerLbl [8]byte
	if _, err := fLbl.Read(headerLbl[:]); err != nil {
		return nil, nil, err
	}
	numLabels := int(binary.BigEndian.Uint32(headerLbl[4:8]))
	if limit > numLabels {
		limit = numLabels
	}

	inputs := make([]map[int]float64, limit)
	labels := make([]int, limit)
	buf := make([]byte, 784)
	for i := 0; i < limit; i++ {
		if _, err := fImg.Read(buf); err != nil {
			return nil, nil, fmt.Errorf("read image data idx=%d: %w", i, err)
		}
		inMap := make(map[int]float64, 784)
		for px := 0; px < 784; px++ {
			inMap[px] = float64(buf[px]) / 255.0
		}
		inputs[i] = inMap

		var lbl [1]byte
		if _, err := fLbl.Read(lbl[:]); err != nil {
			return nil, nil, fmt.Errorf("read label data idx=%d: %w", i, err)
		}
		labels[i] = int(lbl[0])
	}
	return inputs, labels, nil
}

// getRandomSample selects a random subset from the validation set based on the given percentage.
func getRandomSample(inputs []map[int]float64, labels []float64, percent int) ([]map[int]float64, []float64) {
	sampleSize := (len(inputs) * percent) / 100
	if sampleSize < 1 {
		sampleSize = 1
	}
	sampleInputs := make([]map[int]float64, sampleSize)
	sampleLabels := make([]float64, sampleSize)
	perm := rand.Perm(len(inputs))
	for i := 0; i < sampleSize; i++ {
		idx := perm[i]
		sampleInputs[i] = inputs[idx]
		sampleLabels[i] = labels[idx]
	}
	return sampleInputs, sampleLabels
}

// mutateOne applies one random mutation to the given network.
func mutateOne(bp *phase.Phase) {
	/*switch rand.Intn(7) {
	case 0:
		// Add a new neuron and rewire outputs.
		newN := bp.AddRandomNeuron("", "", 1, 50)
		bp.RewireOutputsThroughNewNeuron(newN.ID)
	case 1:
		// Add a random connection.
		bp.AddConnection()
	case 2:
		// Remove a random connection.
		bp.RemoveConnection()
	case 3:
		// Adjust biases.
		bp.AdjustBiases()
	case 4:
		// Adjust connection weights.
		bp.AdjustWeights()
	case 5:
		// Change activation function.
		bp.ChangeActivationFunction()
	case 6:
		// Add several new neurons at once.
		nMax := 10
		nMin := 3
		numNew := rand.Intn(nMax-nMin+1) + nMin
		for i := 0; i < numNew; i++ {
			newN := bp.AddRandomNeuron("", "", 1, 50)
			bp.RewireOutputsThroughNewNeuron(newN.ID)
		}
	}*/

	nMax := 100
	nMin := 30
	numNew := rand.Intn(nMax-nMin+1) + nMin
	for i := 0; i < numNew; i++ {
		newN := bp.AddRandomNeuron("", "", 1, 50)
		bp.RewireOutputsThroughNewNeuron(newN.ID)
	}
}

// saveModel saves the given network as a JSON file in the models directory.
func saveModel(bp *phase.Phase, generation int) {
	if err := os.MkdirAll(modelsDir, os.ModePerm); err != nil {
		fmt.Printf("Error creating models directory: %v\n", err)
		return
	}
	filename := filepath.Join(modelsDir, fmt.Sprintf("best_model_gen%d.json", generation))
	if err := bp.SaveToJSON(filename); err != nil {
		fmt.Printf("Error saving model at generation %d: %v\n", generation, err)
	} else {
		fmt.Printf("Model saved to %s\n", filename)
	}
}
