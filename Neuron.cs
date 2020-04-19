using System;

namespace core_image_recog
{
    public class Neuron
    {
        public double Bias { get; set; }
        public double[] Weights { get; set; }

        public double Output(double[] x)
        {
            return Utils.Activate(Utils.Dot(Weights, x) + Bias);
        }
    }
}