package org.shawn.games.Cortex.DataLoader;

import java.io.DataInputStream;
import java.io.IOException;
import java.util.ArrayList;

public class DataLoader
{
	DataInputStream dataStream;

	public DataLoader(String filePath)
	{
		this.dataStream = new DataInputStream(getClass().getResourceAsStream(filePath));
	}

	public class ConvertedData
	{
		public long occ;
		public byte[] pcs;
		public short score;
		public byte result;
		public byte ksq;
		public byte opp_ksq;
		public byte[] extra;

		public ConvertedData()
		{
			this.pcs = new byte[16];
			this.extra = new byte[3];
		}

		public ArrayList<FeaturePair> toFeatureSet()
		{
			ArrayList<FeaturePair> features = new ArrayList<>();

			long occupancy = this.occ;
			int pieceCount = 0;

			while (occupancy != 0)
			{
				int square = Long.numberOfTrailingZeros(occupancy);
				
				// Two pieces are stored into upper 4 bits and lower 4 bits
				int piece = (this.pcs[pieceCount / 2] >> (4 * (pieceCount & 1))) & 0b1111;
				
				occupancy &= occupancy - 1;
				pieceCount ++;
				
				features.add(new FeaturePair(piece, square));
			}

			return features;
		}
	}

	private static short toLittleEndian(short input)
	{
		return (short) (((input & 0xFF) << 8) | ((input & 0xFF00) >> 8));
	}

	private static long toLittleEndian(long input)
	{
		return Long.reverseBytes(input);
	}

	public ConvertedData load()
	{
		ConvertedData data = new ConvertedData();

		try
		{
			data.occ = toLittleEndian(dataStream.readLong());

			for (int i = 0; i < 16; i++)
			{
				data.pcs[i] = dataStream.readByte();
			}

			data.score = toLittleEndian(dataStream.readShort());
			data.result = dataStream.readByte();
			data.ksq = dataStream.readByte();
			data.opp_ksq = dataStream.readByte();

			for (int i = 0; i < 3; i++)
			{
				data.extra[i] = dataStream.readByte();
			}
		}
		catch (IOException e)
		{
			// TODO Auto-generated catch block
			e.printStackTrace();
			return null;
		}

		return data;
	}
}
