package org.shawn.games.Cortex.LR;

public class LinearLR implements LRScheduler
{
	final double startLR;
	final double gamma;
	final int step;
	int counter;

	public LinearLR(double startLR, double gamma, int step)
	{
		this.startLR = startLR;
		this.gamma = gamma;
		this.step = step;
		this.counter = 0;
	}

	@Override
	public void advance()
	{
		counter++;
	}

	@Override
	public float get()
	{
		double lr = startLR;
		for (int i = 0; i < counter; i += step)
		{
			lr *= gamma;
		}
		return (float) lr;
	}

}
