using System;
using MathNet.Numerics.LinearAlgebra;

namespace core_image_recog {
    public class Utils
    {
        public static double Sigma(double z)
        {
            return 1.0d / (1 + Math.Exp(-z));
        }

        public static double SigmaDash(double z)
        {
            var s = Sigma(z);
            return  s * (1 -s);
        }

        public static Vector<double> Flatten(double[,] twoDArray)
        {
            var result = new double[twoDArray.GetLength(0) * twoDArray.GetLength(1)];
            for (int i = 0; i < twoDArray.GetLength(0); i++) 
            {
                for (int j = 0; j < twoDArray.GetLength(1); j++) 
                {
                    result[i*twoDArray.GetLength(1) + j] = twoDArray[i, j];
                }
            }
            return Vector<double>.Build.Dense(result);
        }

        public static Vector<double> ToVector(byte i)
        {
            var result = new double[10];
            Array.Fill(result, 0d);
            result[i] = 1d;
            return Vector<double>.Build.Dense(result);
        }

        public static double Cost(Vector<double> actual, Vector<double> expected)
        {
            return (actual - expected).Map(a => a*a).Sum();
        }
    }
}