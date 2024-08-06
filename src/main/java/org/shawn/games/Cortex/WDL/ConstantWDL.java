package org.shawn.games.Cortex.WDL;

public class ConstantWDL implements WDLScheduler
{
	float wdlFactor;

	private static final float lerp(final float a, final float b, final float gamma)
	{
		return a * gamma + (1 - gamma) * b;
	}

	public ConstantWDL(float wdlFactor)
	{
		if (wdlFactor < 0 || wdlFactor > 1)
		{
			throw new IllegalArgumentException();
		}

		this.wdlFactor = wdlFactor;
	}

	private static final float sigmoid(float x)
	{
		return (float) (1 / (1 + Math.exp(-x)));
	}

	@Override
	public float getTarget(short score, short result)
	{
		return lerp((float) result / 2f, sigmoid((float) score / 400f), this.wdlFactor);
	}

	@Override
	public void advance()
	{
	}
}
