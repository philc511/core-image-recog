using System;
using MathNet.Numerics.LinearAlgebra;
using MNIST.IO;

namespace core_image_recog
{
    class Program
    {
        static void Main(string[] args)
        {
            var m = Matrix<double>.Build.Random(3, 4);
            Console.WriteLine(m);
            var network = new Network(new int[]{784,15,10});


            var data = FileReaderMNIST.LoadImagesAndLables(
                    "../../data/train-labels-idx1-ubyte.gz",
                    "../../data/train-images-idx3-ubyte.gz");
            double totalCost = 0d;
            int dataSize = 0;
            foreach(var i in data)
            {
                var result = network.Output(Utils.Flatten(i.AsDouble()));
                var expected = Utils.ToVector(i.Label);

                totalCost += Utils.Cost(result, expected);
                // foreach (var d in result) 
                // {
                //     Console.WriteLine(d);
                // }
                //Console.WriteLine(i.Label.ToString());
                //break;
                dataSize++;
                break;
            }
            totalCost = totalCost / (2*dataSize);
            Console.WriteLine(totalCost);
        }




    }
}
