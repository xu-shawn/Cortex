package org.shawn.games.Cortex;

import java.io.*;
import java.util.*;

import org.shawn.games.Cortex.DataLoader.DataLoader;
import org.shawn.games.Cortex.Inputs.Unbucketed768;
import org.shawn.games.Cortex.Network.*;

import org.shawn.games.Cortex.LR.*;

public class Train
{
	public static void main(String[] args)
	{
		DataLoader dl = new DataLoader("/t77.bin");
		SinglePerspective net = new SinglePerspective();

		ArrayList<DataLoader.ConvertedData> testData = new ArrayList<>(10000);
		LRScheduler lr = new ConstantLR(0.02f);

		for (int i = 0; i < 100000; i++)
		{
			testData.add(dl.load());
		}

		for (int i = 0; i < 5000000 - 100000; i++)
		{
			var row = dl.load();
			List<Integer> activatedFeatures = Unbucketed768.featureSetToIndicies(row.toFeatureSet());
			net.backpropagate(activatedFeatures, (float) row.score, lr);

			if (i % 3000 == 0)
			{
				float test_loss = 0;
				for (int j = 0; j < testData.size(); j++)
				{
					var data = testData.get(j);
					float pred = net.forward(Unbucketed768.featureSetToIndicies(data.toFeatureSet()));

					final float deviation = (pred - sigmoid(data.score / 400));
					test_loss += deviation * deviation;
				}
				test_loss /= testData.size();

				System.out.println("Superbatch " + i / 3000 + " Loss: " + test_loss);
			}
		}

		try
		{
			DataOutputStream fileOutput = new DataOutputStream(new FileOutputStream("network.nnue"));
			net.writeQuantized((short)255, (short)64, fileOutput);
		}
		catch (IOException e)
		{
			e.printStackTrace();
		}
	}

	private static final float sigmoid(float x)
	{
		return (float) (1 / (1 + Math.exp(-x)));
	}
}
