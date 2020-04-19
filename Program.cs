using System;

namespace core_image_recog
{
    class Program
    {
        static void Main(string[] args)
        {
            double[] w = {1.0d, 2.0d, 3.0d};
            double[] x = {3.0d, 2.0d, 1.0d};
            Console.WriteLine(Output(w, x, -1));
            Console.WriteLine(Activate(-10000.0d));
            Console.WriteLine(Activate(-0.5d));
            Console.WriteLine(Activate(0.0d));
            Console.WriteLine(Activate(0.5d));
            Console.WriteLine(Activate(10000.0d));
        }

        static double Output(double[] w, double[] x, double b)
        {
            return Activate(Dot(w,x) + b);
        }

        static double Dot(double[] w, double[] x)
        {
            double sum = 0.0d;
            for (int i = 0; i < x.Length; i++)
            {
                sum += w[i] * x[i];
            }
            return sum;
        }

        static double Activate(double z)
        {
            return 1.0d / (1 + Math.Exp(-z));
        }
    }
}
