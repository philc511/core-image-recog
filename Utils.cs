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
    }
}