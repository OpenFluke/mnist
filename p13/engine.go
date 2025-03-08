package main

import (
	"fmt"
	"paragon"
)

func main() {
	// Print version identifier
	fmt.Println("V5-IMPLEMENTATION-13")

	// Define network architecture
	// layerSizes specifies the dimensions of each layer: {Width, Height}
	layerSizes := []struct{ Width, Height int }{
		{28, 28}, // Input layer (e.g., MNIST image)
		{16, 16}, // Hidden layer
		{10, 1},  // Output layer (10 classes, Width=10, Height=1)
	}
	// Activation functions for each layer
	activations := []string{"relu", "relu", "softmax"}
	// Connection type: all layers are partially connected
	fullyConnected := []bool{false, false, false}

	// Create the neural network
	nn := paragon.NewNetwork(layerSizes, activations, fullyConnected)
	nn.Debug = true // Enable debug mode

	// Initialize dummy input (replace with real data in practice)
	// Shape: [numSamples][Height][Width] = [1][28][28]
	inputs := make([][][]float64, 1)
	inputs[0] = make([][]float64, 28)
	for i := range inputs[0] {
		inputs[0][i] = make([]float64, 28)
		inputs[0][i][0] = 1.0 // Simple input value
	}

	// Initialize dummy target (replace with real data in practice)
	// Shape: [numSamples][Height][Width] = [1][1][10], matching output layer
	targets := make([][][]float64, 1)
	targets[0] = make([][]float64, 1)
	targets[0][0] = make([]float64, 10)
	targets[0][0][1] = 1.0 // One-hot encoded for class 1

	// Train the network for 10 epochs with learning rate 0.01
	nn.Train(inputs, targets, 10, 0.01)

	// Add 2 neurons to the last hidden layer (layer index 1)
	nn.AddNeuronsToLayer(1, 2)

	// Train again for 10 epochs
	nn.Train(inputs, targets, 10, 0.01)

	// Perform a forward pass to check the output
	nn.Forward(inputs[0])
	fmt.Println("Output layer values:")
	// Print the output layer neurons (Height=1, Width=10)
	for y := 0; y < nn.Layers[nn.OutputLayer].Height; y++ {
		for x := 0; x < nn.Layers[nn.OutputLayer].Width; x++ {
			fmt.Printf("%.4f ", nn.Layers[nn.OutputLayer].Neurons[y][x].Value)
		}
		fmt.Println()
	}
}
