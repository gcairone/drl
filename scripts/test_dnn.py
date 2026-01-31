import torch
"""
No policy, only test of model training with random data, save/load of model.
"""


from models.base_dnn import BaseDNN

def test_dnn():
    input_dim = 10
    output_dim = 4
    hidden_layers = [32, 32]

    # Crea il modello DNN
    model = BaseDNN(input_dim, output_dim, hidden_layers)

    # Stampa l'architettura del modello
    print("Model architecture:")
    print(model)

    # Test forward pass con dati casuali
    sample_input = torch.randn(1, input_dim)
    sample_output = model(sample_input)
    print("Sample output shape:", sample_output.shape)

    # Salva il modello
    model.save("results/checkpoints/test_dnn_model.pth")
    print("Model saved to test_dnn_model.pth")

    # Carica il modello
    loaded_model = BaseDNN(input_dim, output_dim, hidden_layers)
    loaded_model.load("results/checkpoints/test_dnn_model.pth")
    print("Model loaded from test_dnn_model.pth")

    # Verifica che il modello caricato produca lo stesso output
    loaded_output = loaded_model(sample_input)
    assert torch.allclose(sample_output, loaded_output), "Loaded model output does not match original!"
    print("Loaded model output matches original.")

def test_training():
    # Test semplice di training con dati casuali
    input_dim = 10
    output_dim = 4
    hidden_layers = [32, 32]

    model = BaseDNN(input_dim, output_dim, hidden_layers)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Dati casuali
    inputs = torch.randn(100, input_dim)
    targets = torch.randn(100, output_dim)

    # Training loop semplice
    for epoch in range(10):
        model.train()
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

    # Salva il modello
    model.save("checkpoints/test_training_model.pth")
    print("Model saved to test_training_model.pth")


if __name__ == "__main__":
    test_dnn()
    test_training()