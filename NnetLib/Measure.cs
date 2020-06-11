using System;
using MathNet.Numerics;
using MathNet.Numerics.LinearAlgebra;

namespace NnetLib
{
    public class Measure
    {
        public double Score(int[] expected, int[] actual)
        {
            var count = 0;
            for (int i = 0; i < expected.Length; i++) 
            {
                if (expected[i] == actual[i])
                {
                    count++;
                }
            }
            return (double)count/expected.Length;
        }
    }
}