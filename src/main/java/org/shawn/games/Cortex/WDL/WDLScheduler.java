package org.shawn.games.Cortex.WDL;

public interface WDLScheduler
{	
	public float getTarget(short score, short result);
	
	public void advance();
}
