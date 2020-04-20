using System;

namespace core_image_recog
{
    public class Neuron
    {
        private int size;

        public Neuron(int size)
        {
            this.size = size;
            Bias = 0.0;
            Weights = new double[size];
            Array.Fill(Weights, 1.0d);
        }
        public double Bias { get; set; }
        public double[] Weights { get; set; }

        public double Output(double[] x)
        {
            return Utils.Activate(Utils.Dot(Weights, x) + Bias);
        }
    }
}