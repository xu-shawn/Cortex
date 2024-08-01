package org.shawn.games.Cortex;

import static org.junit.Assert.*;

import org.junit.Test;

import org.shawn.games.Cortex.DataLoader.DataLoader;
import org.shawn.games.Cortex.DataLoader.FeaturePair;

public class DataLoaderTest
{
	DataLoader data;

	public DataLoaderTest()
	{
	}

	@Test
	public void testFeatureSet()
	{
		data = new DataLoader("/t77.bin");

		var convertedData = data.load();

		// @formatter:off
		int[] expected = new int[]
		{
				4,  2,  3,  5,  6,  3,  2,  4,
				1,  1,  1,  1,  1,  1,  1,  1,
				0,  0,  0,  0,  0,  0,  0,  0,
				0,  0,  0,  0,  0,  0,  0,  0,
				0,  0,  0,  0,  0,  0,  0,  0,
				0,  0,  0,  0,  0,  0,  0,  0,
				9,  9,  9,  9,  9,  9,  9,  9,
				12, 10, 11, 13, 14, 11, 10, 12
		};
		// @formatter:on

		for (FeaturePair x : convertedData.toFeatureSet())
		{
			assertEquals(expected[x.square()], x.piece() + 1);
		}

		assertEquals(expected[convertedData.ksq], 6);
		assertEquals(expected[convertedData.opp_ksq ^ 0b111000], 14);

		data.load();
		convertedData = data.load();

		expected = new int[] { 4, 2, 3, 5, 6, 3, 2, 4, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
				0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 9, 0, 0, 0, 0, 0, 0, 9, 0, 9, 9, 9, 9, 9, 9, 12, 10, 11, 13, 14, 11,
				10, 12 };

		for (FeaturePair x : convertedData.toFeatureSet())
		{
			assertEquals(expected[x.square()], x.piece() + 1);
		}

		assertEquals(expected[convertedData.ksq], 6);
		assertEquals(expected[convertedData.opp_ksq ^ 0b111000], 14);

		convertedData = data.load();

		expected = new int[] { 4, 2, 3, 5, 6, 3, 2, 4, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
				0, 0, 0, 0, 0, 0, 9, 0, 0, 0, 0, 0, 10, 0, 0, 0, 0, 0, 9, 9, 9, 9, 0, 9, 9, 9, 12, 0, 11, 13, 14, 11,
				10, 12 };

		for (FeaturePair x : convertedData.toFeatureSet())
		{
			assertEquals(expected[x.square()], x.piece() + 1);
		}

		assertEquals(expected[convertedData.ksq], 6);
		assertEquals(expected[convertedData.opp_ksq ^ 0b111000], 14);
	}

	@SuppressWarnings("unused")
	private static void printBoardFromFeatureSet(DataLoader.ConvertedData convertedData)
	{
		int[] board = new int[64];

		for (FeaturePair x : convertedData.toFeatureSet())
		{
			board[x.square()] = x.piece() + 1;
		}

		for (int i = 0; i < 7; i++)
		{
			for (int j = 0; j < 8; j++)
			{
				System.out.print(board[i * 8 + j] + "  ");
			}

			System.out.println();
		}

		for (int j = 0; j < 8; j++)
		{
			System.out.print(board[7 * 8 + j] + " ");
		}
	}

}
