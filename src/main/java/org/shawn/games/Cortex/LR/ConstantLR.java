package org.shawn.games.Cortex.LR;

public class ConstantLR implements LRScheduler
{
	final float lr;

	public ConstantLR(float lr)
	{
		this.lr = lr;
	}

	@Override
	public void advance()
	{
	}

	@Override
	public float get()
	{
		return lr;
	}
}
