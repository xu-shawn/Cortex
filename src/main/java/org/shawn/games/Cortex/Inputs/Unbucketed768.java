package org.shawn.games.Cortex.Inputs;

import java.util.*;
import java.util.stream.Collectors;

import org.shawn.games.Cortex.DataLoader.FeaturePair;

public class Unbucketed768
{
	public static int featurePairToIndex(FeaturePair fp)
	{
		return (fp.piece() > 5 ? fp.piece() - 2 : fp.piece()) * 64 + fp.square();
	}

	public static List<Integer> featureSetToIndicies(ArrayList<FeaturePair> fp)
	{
		return fp.stream().map(Unbucketed768::featurePairToIndex).collect(Collectors.toList());
	}
}
