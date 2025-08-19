[![codecov](https://codecov.io/gh/takanori-ugai/D2J/graph/badge.svg?token=5EAXZ3ES0N)](https://codecov.io/gh/takanori-ugai/D2J)

# Dive into Deep Learning - Kotlin Implementation

This repository contains a Kotlin implementation of various deep learning models and concepts, primarily based on the book "Dive into Deep Learning". The implementations use the Deep Java Library (DJL) for tensor computations and neural network layers.

## Description

This project is intended for educational purposes, providing a way to study and experiment with deep learning models in a Kotlin environment. Each model is implemented in a clear and concise way, following the structure of the "Dive into Deep Learning" book.

## Models Implemented

The repository includes implementations of the following models:

*   **Classic Networks:**
    *   `AlexNet`
    *   `LeNet`
*   **Recurrent Neural Networks (RNNs):**
    *   `RNN` from scratch and concise implementations
    *   `GRU`
    *   `LSTM`
    *   `DeepRnn`
    *   `BiRnn`
*   **Attention Mechanisms & Transformers:**
    *   `SelfAttention`
    *   `MultiheadAttention`
    *   `Transformer`
    *   `Seq2Seq` with Attention
*   **Computer Vision:**
    *   `VisionTransformer` (ViT)
    *   `BatchNorm`
    *   `DenseBlock` (used in DenseNet)
    *   `Residual` blocks (used in ResNet)
*   **Natural Language Processing (NLP):**
    *   `WordEmbedding`
    *   `Language` models
    *   `MachineTranslation`
    *   `SeqDataLoader`

## Dependencies

*   **Kotlin:** The core programming language.
*   **Deep Java Library (DJL):** For deep learning operations. The project uses the MXNet engine.
*   **lets-plot:** For creating plots and visualizations.
*   **Gradle:** For dependency management and building the project.

## How to Run

Each model or example is contained in its own file with a `main` function. To run a specific example, you need to configure the `build.gradle.kts` file.

1.  **Open the `build.gradle.kts` file** in the root of the project.
2.  **Locate the `application` block:**
    ```kotlin
    application {
        mainClass.set("jp.live.ugai.d2j.BatchNorm2Kt")
    }
    ```
3.  **Change the `mainClass`** to the file you want to run. For example, to run the Transformer example, you would change it to `jp.live.ugai.d2j.TransformerKt`. Note that the class name for a file named `MyFile.kt` will be `MyFileKt`.
4.  **Run the code** from your terminal using Gradle:
    ```bash
    ./gradlew run
    ```
    Alternatively, you can use an IDE like IntelliJ IDEA to open the project and run the `main` function of any file directly.

## License

This project is licensed under the Apache License 2.0. See the [LICENSE](LICENSE) file for details.
