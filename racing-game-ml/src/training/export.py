"""
Model Export Utilities for Racing Game ML

Export trained models to formats suitable for frontend inference:
- ONNX format (for ONNX.js in browser)
- JSON format (for custom inference)
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from stable_baselines3 import PPO


def export_to_onnx(
    model_path: str,
    output_path: str,
    opset_version: int = 11,
    verbose: bool = True,
) -> str:
    """
    Export a trained PPO model to ONNX format.

    The exported model takes observations as input and outputs
    action logits (before softmax).

    Args:
        model_path: Path to the trained SB3 model
        output_path: Path for the output ONNX file
        opset_version: ONNX opset version
        verbose: Print export information

    Returns:
        Path to the exported ONNX file
    """
    try:
        import onnx
        import onnxruntime
    except ImportError:
        raise ImportError(
            "ONNX export requires 'onnx' and 'onnxruntime' packages. "
            "Install with: pip install onnx onnxruntime"
        )

    # Load the model
    model = PPO.load(model_path)

    # Extract the policy network
    policy = model.policy

    # Get observation space dimensions
    obs_dim = model.observation_space.shape[0]
    n_actions = model.action_space.n

    if verbose:
        print(f"Model loaded: {model_path}")
        print(f"Observation dim: {obs_dim}")
        print(f"Action space: {n_actions} actions")

    # Create a wrapper module for ONNX export
    class PolicyWrapper(nn.Module):
        """Wrapper to export just the action selection part"""

        def __init__(self, policy):
            super().__init__()
            # Extract the MLP extractor and action net
            self.features_extractor = policy.features_extractor
            self.mlp_extractor = policy.mlp_extractor
            self.action_net = policy.action_net

        def forward(self, obs: torch.Tensor) -> torch.Tensor:
            """
            Forward pass: observation -> action probabilities

            Args:
                obs: Observation tensor [batch, obs_dim]

            Returns:
                Action probabilities [batch, n_actions]
            """
            features = self.features_extractor(obs)
            latent_pi, _ = self.mlp_extractor(features)
            action_logits = self.action_net(latent_pi)
            action_probs = torch.softmax(action_logits, dim=-1)
            return action_probs

    # Create wrapper and set to eval mode
    wrapper = PolicyWrapper(policy)
    wrapper.eval()

    # Create dummy input
    dummy_input = torch.randn(1, obs_dim)

    # Ensure output directory exists
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Export to ONNX
    if verbose:
        print(f"\nExporting to ONNX: {output_path}")

    torch.onnx.export(
        wrapper,
        dummy_input,
        str(output_path),
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=["observation"],
        output_names=["action_probs"],
        dynamic_axes={
            "observation": {0: "batch_size"},
            "action_probs": {0: "batch_size"},
        },
    )

    if verbose:
        print("ONNX export successful!")

    # Verify the exported model
    onnx_model = onnx.load(str(output_path))
    onnx.checker.check_model(onnx_model)

    if verbose:
        print("ONNX model verification passed!")

    # Test with ONNX Runtime
    ort_session = onnxruntime.InferenceSession(str(output_path))
    test_input = np.random.randn(1, obs_dim).astype(np.float32)
    ort_outputs = ort_session.run(None, {"observation": test_input})

    if verbose:
        print(f"\nTest inference:")
        print(f"  Input shape: {test_input.shape}")
        print(f"  Output shape: {ort_outputs[0].shape}")
        print(f"  Output (action probs): {ort_outputs[0][0]}")
        print(f"  Selected action: {np.argmax(ort_outputs[0][0])}")

    return str(output_path)


def export_to_json(
    model_path: str,
    output_path: str,
    verbose: bool = True,
) -> str:
    """
    Export model weights to JSON format for custom inference.

    This creates a JSON file with all network weights that can be
    loaded in JavaScript for custom inference without ONNX.js.

    Args:
        model_path: Path to the trained SB3 model
        output_path: Path for the output JSON file
        verbose: Print export information

    Returns:
        Path to the exported JSON file
    """
    # Load the model
    model = PPO.load(model_path)
    policy = model.policy

    if verbose:
        print(f"Model loaded: {model_path}")

    # Extract weights from each layer
    weights: Dict[str, Any] = {
        "meta": {
            "obs_dim": model.observation_space.shape[0],
            "n_actions": model.action_space.n,
            "format": "stable_baselines3_ppo",
        },
        "layers": [],
    }

    def extract_linear_layer(name: str, layer: nn.Linear) -> Dict[str, Any]:
        """Extract weights and bias from a linear layer"""
        return {
            "name": name,
            "type": "linear",
            "weight": layer.weight.detach().cpu().numpy().tolist(),
            "bias": layer.bias.detach().cpu().numpy().tolist(),
            "in_features": layer.in_features,
            "out_features": layer.out_features,
        }

    # Extract MLP extractor layers
    mlp = policy.mlp_extractor

    # Policy network layers
    for i, layer in enumerate(mlp.policy_net):
        if isinstance(layer, nn.Linear):
            weights["layers"].append(
                extract_linear_layer(f"policy_net.{i}", layer)
            )

    # Action network (final layer)
    weights["layers"].append(
        extract_linear_layer("action_net", policy.action_net)
    )

    # Also include value network for completeness
    weights["value_layers"] = []
    for i, layer in enumerate(mlp.value_net):
        if isinstance(layer, nn.Linear):
            weights["value_layers"].append(
                extract_linear_layer(f"value_net.{i}", layer)
            )
    weights["value_layers"].append(
        extract_linear_layer("value_net_output", policy.value_net)
    )

    # Save to JSON
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(weights, f)

    if verbose:
        file_size = output_path.stat().st_size / 1024
        print(f"\nJSON export successful!")
        print(f"  Output: {output_path}")
        print(f"  File size: {file_size:.1f} KB")
        print(f"  Layers: {len(weights['layers'])}")

    return str(output_path)


def create_inference_code(
    model_path: str,
    output_dir: str,
    verbose: bool = True,
) -> Tuple[str, str]:
    """
    Create JavaScript inference code along with the model weights.

    Args:
        model_path: Path to the trained SB3 model
        output_dir: Directory for output files
        verbose: Print export information

    Returns:
        Tuple of (weights_path, js_code_path)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Export weights to JSON
    weights_path = export_to_json(
        model_path,
        str(output_dir / "model_weights.json"),
        verbose=verbose,
    )

    # Create inference code
    js_code = '''/**
 * Racing Game ML Model Inference (Browser)
 *
 * Load and run the trained PPO model for CPU car control.
 * Uses JSON weights exported from Python training.
 */

class RacingMLModel {
    constructor() {
        this.weights = null;
        this.loaded = false;
    }

    /**
     * Load model weights from JSON file
     * @param {string} weightsUrl - URL to model_weights.json
     */
    async load(weightsUrl) {
        const response = await fetch(weightsUrl);
        this.weights = await response.json();
        this.loaded = true;
        console.log(`Model loaded: ${this.weights.meta.obs_dim} inputs, ${this.weights.meta.n_actions} actions`);
    }

    /**
     * Apply a linear layer
     * @param {number[]} input - Input vector
     * @param {Object} layer - Layer weights
     * @returns {number[]} Output vector
     */
    linearLayer(input, layer) {
        const output = new Array(layer.out_features).fill(0);
        for (let i = 0; i < layer.out_features; i++) {
            output[i] = layer.bias[i];
            for (let j = 0; j < layer.in_features; j++) {
                output[i] += layer.weight[i][j] * input[j];
            }
        }
        return output;
    }

    /**
     * Apply tanh activation
     * @param {number[]} x - Input vector
     * @returns {number[]} Output vector
     */
    tanh(x) {
        return x.map(v => Math.tanh(v));
    }

    /**
     * Apply softmax to get probabilities
     * @param {number[]} x - Input logits
     * @returns {number[]} Probabilities
     */
    softmax(x) {
        const max = Math.max(...x);
        const exp = x.map(v => Math.exp(v - max));
        const sum = exp.reduce((a, b) => a + b, 0);
        return exp.map(v => v / sum);
    }

    /**
     * Get action from observation
     * @param {number[]} observation - Environment observation (15 features)
     * @param {boolean} deterministic - If true, always select best action
     * @returns {number} Selected action (0-8)
     */
    predict(observation, deterministic = true) {
        if (!this.loaded) {
            throw new Error("Model not loaded. Call load() first.");
        }

        // Forward pass through policy network
        let x = observation;

        // Apply each layer with tanh activation
        for (let i = 0; i < this.weights.layers.length - 1; i++) {
            x = this.linearLayer(x, this.weights.layers[i]);
            x = this.tanh(x);
        }

        // Final layer (action logits)
        const logits = this.linearLayer(x, this.weights.layers[this.weights.layers.length - 1]);

        if (deterministic) {
            // Select action with highest logit
            return logits.indexOf(Math.max(...logits));
        } else {
            // Sample from probability distribution
            const probs = this.softmax(logits);
            const random = Math.random();
            let cumsum = 0;
            for (let i = 0; i < probs.length; i++) {
                cumsum += probs[i];
                if (random < cumsum) {
                    return i;
                }
            }
            return probs.length - 1;
        }
    }

    /**
     * Get action probabilities
     * @param {number[]} observation - Environment observation
     * @returns {number[]} Action probabilities
     */
    getActionProbabilities(observation) {
        if (!this.loaded) {
            throw new Error("Model not loaded. Call load() first.");
        }

        let x = observation;
        for (let i = 0; i < this.weights.layers.length - 1; i++) {
            x = this.linearLayer(x, this.weights.layers[i]);
            x = this.tanh(x);
        }
        const logits = this.linearLayer(x, this.weights.layers[this.weights.layers.length - 1]);
        return this.softmax(logits);
    }
}

// Action mapping (matching Python environment)
const ACTIONS = {
    0: { accelerate: false, brake: false, steerLeft: false, steerRight: false }, // Coast
    1: { accelerate: true, brake: false, steerLeft: false, steerRight: false },  // Accelerate
    2: { accelerate: false, brake: true, steerLeft: false, steerRight: false },  // Brake
    3: { accelerate: false, brake: false, steerLeft: true, steerRight: false },  // Steer left
    4: { accelerate: false, brake: false, steerLeft: false, steerRight: true },  // Steer right
    5: { accelerate: true, brake: false, steerLeft: true, steerRight: false },   // Accel + left
    6: { accelerate: true, brake: false, steerLeft: false, steerRight: true },   // Accel + right
    7: { accelerate: false, brake: true, steerLeft: true, steerRight: false },   // Brake + left
    8: { accelerate: false, brake: true, steerLeft: false, steerRight: true },   // Brake + right
};

/**
 * Convert action index to input controls
 * @param {number} action - Action index (0-8)
 * @returns {Object} Input controls
 */
function actionToInput(action) {
    return ACTIONS[action] || ACTIONS[0];
}

// Export for use in game
if (typeof module !== 'undefined') {
    module.exports = { RacingMLModel, ACTIONS, actionToInput };
}
'''

    js_path = output_dir / "inference.js"
    with open(js_path, "w") as f:
        f.write(js_code)

    if verbose:
        print(f"\nJavaScript inference code created: {js_path}")

    return weights_path, str(js_path)


def export_model(
    model_path: str,
    output_dir: str,
    formats: List[str] = ["onnx", "json"],
    verbose: bool = True,
) -> Dict[str, str]:
    """
    Export model to multiple formats.

    Args:
        model_path: Path to trained model
        output_dir: Output directory
        formats: List of formats to export ("onnx", "json")
        verbose: Print information

    Returns:
        Dictionary mapping format to output path
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {}

    if "onnx" in formats:
        try:
            onnx_path = export_to_onnx(
                model_path,
                str(output_dir / "model.onnx"),
                verbose=verbose,
            )
            results["onnx"] = onnx_path
        except ImportError as e:
            print(f"Warning: ONNX export skipped - {e}")

    if "json" in formats:
        json_path, js_path = create_inference_code(
            model_path,
            str(output_dir),
            verbose=verbose,
        )
        results["json"] = json_path
        results["js"] = js_path

    return results


# CLI entry point
def main():
    """CLI entry point for model export"""
    import argparse

    parser = argparse.ArgumentParser(description="Export trained models for frontend use")
    parser.add_argument("model_path", type=str, help="Path to trained model")
    parser.add_argument("-o", "--output", type=str, default="./export",
                        help="Output directory")
    parser.add_argument("--format", type=str, nargs="+",
                        choices=["onnx", "json", "all"], default=["all"],
                        help="Export format(s)")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Verbose output")

    args = parser.parse_args()

    formats = ["onnx", "json"] if "all" in args.format else args.format

    results = export_model(
        args.model_path,
        args.output,
        formats=formats,
        verbose=args.verbose or True,
    )

    print("\n" + "=" * 40)
    print("Export complete!")
    print("=" * 40)
    for fmt, path in results.items():
        print(f"  {fmt}: {path}")


if __name__ == "__main__":
    main()
