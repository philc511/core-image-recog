using System;
using MathNet.Numerics.LinearAlgebra;
using MNIST.IO;

namespace core_image_recog
{
    class Program
    {
        static void Main(string[] args)
        {
            Log("Starting");
            var network = new Network(new int[] { 784, 15, 10 });

            var data = FileReaderMNIST.LoadImagesAndLables(
                    "../../data/train-labels-idx1-ubyte.gz",
                    "../../data/train-images-idx3-ubyte.gz");
            double totalCost = 0d;

            var inputs = new Vector<double>[60000];
            var expected = new Vector<double>[60000];
            int n = 0;
            foreach (var x in data)
            {
                inputs[n] = Utils.Flatten(x.AsDouble());
                expected[n] = Utils.ToVector(x.Label);
                n++;
            }
            Log("Read data in");

            Log("Initial cost=" + GetCost(inputs, expected, network));

            int batchSize = 1000;
            for (int j = 0; j < 50000; j++)
            {
                var x = inputs[j];
                var result = network.FeedForward(x);
                var y = expected[j];

                network.BackProp(y);
                network.AdjustDeltaSums(x);
            }
            network.GradDesc(0.01, batchSize);
            Log("After 100=" + GetCost(inputs, expected, network));
        }

        private static double GetCost(Vector<double>[] inputs, Vector<double>[] expected, Network network)
        {
            var cost = 0d;
            var size = inputs.Length;
            for (int i = 0; i < size; i++)
            {
                var al = network.FeedForward(inputs[i]);
                cost += Utils.Cost(al, expected[i]);
            }
            return cost / 2 / size;
        }

        private static void Log(string msg)
        {
            Console.WriteLine(DateTime.Now.ToLongTimeString() + ": " + msg);
        }
    }
}
