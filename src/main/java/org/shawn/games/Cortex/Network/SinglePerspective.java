package org.shawn.games.Cortex.Network;

import java.io.DataOutputStream;
import java.io.IOException;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

import org.shawn.games.Cortex.LR.LRScheduler;

public class SinglePerspective
{
	// 768 -> X -> 1
	// SCReLU Activated
	// Mean Squared Error

	private final static int HIDDEN_LAYER = 1024;
	private final static float MAX_WEIGHTS = 1.98f;

	float[][] ftWeights;
	float[] ftBiases;
	float[] outputWeights;
	float outputBias;

	boolean activated[];
	float hiddenLayer[];
	float output;

	public SinglePerspective()
	{
		this.ftWeights = new float[768][HIDDEN_LAYER];
		this.ftBiases = new float[HIDDEN_LAYER];
		this.outputWeights = new float[HIDDEN_LAYER];
		this.outputBias = 0;

		for (int i = 0; i < 768; i++)
		{
			xavierInitialize(this.ftWeights[i], 768, HIDDEN_LAYER);
		}

		xavierInitialize(this.ftBiases, HIDDEN_LAYER, 1);

		this.activated = new boolean[768];
		this.hiddenLayer = new float[HIDDEN_LAYER];
		this.output = 0;
	}

	private static final void xavierInitialize(float[] weights, int in, int out)
	{
		float limit = (float) Math.sqrt(6.0 / (in + out));
		Random r = new Random();

		for (int i = 0; i < weights.length; ++i)
		{
			weights[i] = (r.nextFloat() * 2 * limit) - limit;
			weights[i] = Math.max(Math.min(weights[i], MAX_WEIGHTS), -MAX_WEIGHTS);
		}
	}

	private static final float screlu(final float x)
	{
		float v = Math.max(Math.min(x, 1), 0);
		return v * v;
	}

	private static final float screluDerivative(final float x)
	{
		return (x > 0 && x < 1) ? 2 * x : 0;
	}

	private static final float sigmoid(final float x)
	{
		return (float) (1 / (1 + Math.exp(-x)));
	}

	private static final float sigmoidDerivative(final float x)
	{
		return sigmoid(x) * (1 - sigmoid(x));
	}

	public float forward(List<Integer> activatedFeatures)
	{
		System.arraycopy(this.ftBiases, 0, this.hiddenLayer, 0, HIDDEN_LAYER);
		Arrays.fill(activated, false);

		for (int index : activatedFeatures)
		{
			activated[index] = true;
			for (int i = 0; i < HIDDEN_LAYER; i++)
			{
				this.hiddenLayer[i] += this.ftWeights[index][i];
			}
		}

		for (int i = 0; i < HIDDEN_LAYER; i++)
		{
			this.hiddenLayer[i] = screlu(this.hiddenLayer[i]);
		}

		this.output = outputBias;

		for (int i = 0; i < HIDDEN_LAYER; i++)
		{
			this.output += this.hiddenLayer[i] * this.outputWeights[i];
		}

		this.output = sigmoid(this.output);

		return this.output;
	}

	public float forward2(List<Integer> activatedFeatures)
	{
		System.arraycopy(this.ftBiases, 0, this.hiddenLayer, 0, HIDDEN_LAYER);
		Arrays.fill(activated, false);

		for (int index : activatedFeatures)
		{
			activated[index] = true;
			for (int i = 0; i < HIDDEN_LAYER; i++)
			{
				this.hiddenLayer[i] += this.ftWeights[index][i];
			}
		}

		for (int i = 0; i < HIDDEN_LAYER; i++)
		{
			this.hiddenLayer[i] = screlu(this.hiddenLayer[i]);
		}

		this.output = outputBias;

		for (int i = 0; i < HIDDEN_LAYER; i++)
		{
			this.output += this.hiddenLayer[i] * this.outputWeights[i];
		}

		return this.output;
	}

	public float backpropagate(List<Integer> activatedFeatures, float target, LRScheduler lrScheduler)
	{
		forward(activatedFeatures);

		float lr = lrScheduler.get();

		// Mean Squared Error
		float delta = (this.output - target);
		float error = delta * delta;

		float outputBiasGradient = 2 * delta * sigmoidDerivative(this.output);

		float[] outputWeightGradient = new float[HIDDEN_LAYER];
		float[] ftBiasGradient = new float[HIDDEN_LAYER];
		float[][] ftWeightGradient = new float[768][HIDDEN_LAYER];

		for (int i = 0; i < HIDDEN_LAYER; i++)
		{
			outputWeightGradient[i] = outputBiasGradient * this.hiddenLayer[i];
		}

		for (int i = 0; i < HIDDEN_LAYER; i++)
		{
			ftBiasGradient[i] = screluDerivative(this.hiddenLayer[i]) * this.outputWeights[i] * outputBiasGradient;
		}

		for (int i = 0; i < 768; i++)
		{
			if (this.activated[i])
			{
				for (int j = 0; j < HIDDEN_LAYER; j++)
				{
					ftWeightGradient[i][j] = ftBiasGradient[j];
				}
			}
		}

		this.outputBias -= lr * outputBiasGradient;

		for (int i = 0; i < HIDDEN_LAYER; i++)
		{
			this.outputWeights[i] -= lr * outputWeightGradient[i];

			this.outputWeights[i] = Math.max(Math.min(this.outputWeights[i], MAX_WEIGHTS), -MAX_WEIGHTS);

			this.ftBiases[i] -= lr * ftBiasGradient[i];
		}

		for (int i = 0; i < 768; i++)
		{
			// Optimization: Gradient is 0 if this.activated[i] = false
			if (this.activated[i])
			{
				for (int j = 0; j < HIDDEN_LAYER; j++)
				{
					this.ftWeights[i][j] -= lr * ftWeightGradient[i][j];
					this.ftWeights[i][j] = Math.max(Math.min(this.ftWeights[i][j], MAX_WEIGHTS), -MAX_WEIGHTS);
				}
			}
		}

		return error;
	}

	private static short quantizeAndReverseEndianness(final float v, final short quantizationFactor)
	{
		short quantized = (short) (v * quantizationFactor);
		return Short.reverseBytes(quantized);
	}

	public void writeQuantized(final short QA, final short QB, DataOutputStream file) throws IOException
	{
		for (int i = 0; i < 768; i++)
		{
			for (int j = 0; j < HIDDEN_LAYER; j++)
			{
				file.writeShort(quantizeAndReverseEndianness(this.ftWeights[i][j], QA));
			}
		}

		for (int i = 0; i < HIDDEN_LAYER; i++)
		{
			file.writeShort(quantizeAndReverseEndianness(this.ftBiases[i], QA));
		}

		for (int i = 0; i < HIDDEN_LAYER; i++)
		{
			file.writeShort(quantizeAndReverseEndianness(this.outputWeights[i], QB));
		}

		file.writeShort(quantizeAndReverseEndianness(this.outputBias, (short) (QA * QB)));
	}
}
