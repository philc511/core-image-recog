using System;

namespace core_image_recog
{
    public class Neuron
    {
        private int size;

        public Neuron(int size)
        {
            this.size = size;
            Bias = -1.0d;
            Weights = new double[size];
            Array.Fill(Weights, 1.0d/size);
        }
        public double Bias { get; set; }
        public double[] Weights { get; set; }

        public double Output(double[] x)
        {
            var d = Utils.Activate(Utils.Dot(Weights, x) + Bias);
            return d;
        }
    }
}