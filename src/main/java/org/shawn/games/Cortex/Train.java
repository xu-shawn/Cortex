package org.shawn.games.Cortex;

import java.io.*;
import java.util.*;

import org.shawn.games.Cortex.DataLoader.DataLoader;
import org.shawn.games.Cortex.Inputs.Unbucketed768;
import org.shawn.games.Cortex.Network.*;
import org.shawn.games.Cortex.WDL.*;
import org.shawn.games.Cortex.LR.*;

public class Train
{
	public static void main(String[] args)
	{
		DataLoader dl2 = new DataLoader("/test.bin");

		DataLoader.ConvertedData startpos = dl2.load();

		SinglePerspective net = new SinglePerspective();
		
		WDLScheduler wdl = new ConstantWDL(0.75f);

		for (int epoch = 0; epoch < 10; epoch++)
		{
			DataLoader dl = new DataLoader("/integral.bin");
			ArrayList<DataLoader.ConvertedData> testData = new ArrayList<>(10000);
			LRScheduler lr = new ConstantLR(0.001f - epoch * 0.00005f);

			for (int i = 0; i < 100000; i++)
			{
				testData.add(dl.load());
			}

			for (int i = 0; i < 7000000 - 100000; i++)
			{
				var row = dl.load();
				if (row == null)
				{
					break;
				}
				List<Integer> activatedFeatures = Unbucketed768.featureSetToIndicies(row.toFeatureSet());
				net.backpropagate(activatedFeatures, wdl.getTarget(row.score, row.result), lr);

				if (i % 3000 == 0)
				{
					float test_loss = 0;
					for (int j = 0; j < testData.size(); j++)
					{
						var data = testData.get(j);
						float pred = net.forward(Unbucketed768.featureSetToIndicies(data.toFeatureSet()));

						final float deviation = (pred - wdl.getTarget(data.score, data.result));
						test_loss += deviation * deviation;
					}
					test_loss /= testData.size();

					System.out.println(400 * net.forward2(Unbucketed768.featureSetToIndicies(startpos.toFeatureSet())));

					System.out.println("Epoch " + epoch + " Superbatch " + i / 3000 + " Loss: " + test_loss);
				}
			}
			
			lr.advance();
			wdl.advance();
		}

		try
		{
			DataOutputStream fileOutput = new DataOutputStream(new FileOutputStream("network.nnue"));
			net.writeQuantized((short) 255, (short) 64, fileOutput);
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
