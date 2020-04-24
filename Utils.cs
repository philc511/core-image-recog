using System;

namespace core_image_recog {
    public class Utils
    {
        // TODO should make this a function injected into a Neuron
        public static double Activate(double z)
        {
            return 1.0d / (1 + Math.Exp(-z));
        }

        public static double Dot(double[] w, double[] x)
        {
            double sum = 0.0d;
            for (int i = 0; i < x.Length; i++)
            {
                sum += w[i] * x[i];
            }
            return sum;
        }

        public static double[] Flatten(double[,] twoDArray)
        {
            var result = new double[twoDArray.GetLength(0) * twoDArray.GetLength(1)];
            for (int i = 0; i < twoDArray.GetLength(0); i++) 
            {
                for (int j = 0; j < twoDArray.GetLength(1); j++) 
                {
                    result[i*twoDArray.GetLength(1) + j] = twoDArray[i, j];
                }
            }
            return result;
        }

        public static double[] ToVector(byte i)
        {
            var result = new double[10];
            Array.Fill(result, 0d);
            result[i] = 1d;
            return result;
        }

        public static double Cost(double[] actual, double[] expected)
        {
            var cost = 0d;
            for (int i = 0; i < actual.Length; i++) 
            {
                double d = actual[i] - expected[1];
                cost += d*d;
            }
            return cost;
        }
    }
}