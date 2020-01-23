package uk.ac.cam.mgm52.cnn;

import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.function.BinaryOperator;
import java.util.stream.Collectors;
import java.util.stream.IntStream;


/** To assist in operating on arrays*/
public final class ArrayUtils {

    private ArrayUtils(){}

    //Apply this binary operator between all items of arr and arr2. In the case of length discrepancy, elements from longer array are taken.
    public static int[] applyAll(int[] arr, int[] arr2, BinaryOperator<Integer> op){
        int[] max;
        int[] min;

        if(arr.length>arr2.length){
            max = arr;
            min = arr2;
        }
        else{
            max = arr2;
            min = arr;
        }

        int[] result = new int[max.length];

        //Apply function
        for(int i = 0; i < min.length; i++){
            //This cast from Integer -> int may be introducing overhead, I don't think it's significant though.
            result[i] = op.apply(arr[i], arr2[i]);
        }

        //Account for array length discrepancy by copyin over values from max
        System.arraycopy(max, min.length, result, min.length, max.length-min.length);

        return result;
    }



    //Could avoid having to create two applyAll functions if I used objects instead of primitives,
    public static double[] applyAll(double[] arr, double[] arr2, BinaryOperator<Double> op){
        double[] max;
        double[] min;

        if(arr.length>arr2.length){
            max = arr;
            min = arr2;
        }
        else{
            max = arr2;
            min = arr;
        }

        double[] result = new double[max.length];

        //Apply function
        for(int i = 0; i < min.length; i++){
            //This cast from Integer -> int may be introducing overhead, I don't think it's significant though.
            result[i] = op.apply(arr[i], arr2[i]);
        }

        //Account for array length discrepancy
        for(int i = min.length; i < max.length; i++){
            result[i] = max[i];
        }

        return result;
    }


    public static int[] addAll(int[] arr, int[] arr2){return applyAll(arr, arr2, (a,b)->a+b);}
    public static int[] addAll(int[] arr, int f) {return Arrays.stream(arr).map(a->a+f).toArray();}
    public static int[] subtractAll(int[] arr, int[] arr2){return applyAll(arr, arr2, (a,b)->a-b);}
    public static int[] divideAll(int[] arr, int[] arr2){return applyAll(arr, arr2, (a,b)->a/b);}
    public static int[] multAll(int[] arr, int[] arr2){return applyAll(arr, arr2, (a,b)->a*b);}

    public static double[] addAll(double[] arr, double[] arr2){return applyAll(arr, arr2, (a,b)->a+b);}
    public static double[] multAll(double[] arr, double[] arr2){return applyAll(arr, arr2, (a,b)->a*b);}
    public static double[] multAll(double[] arr, double f) {return Arrays.stream(arr).map(a->a*f).toArray();}
    public static double[] divideAll(double[] arr, double f) {return Arrays.stream(arr).map(a->a/f).toArray();}




    public static int[] appendValue(int[] arr, int val){
        int[] newArray = Arrays.copyOf(arr, arr.length+1);
        newArray[newArray.length - 1] = val;

        return newArray;
    }

    public static <T> T[] appendValue(T[] arr, T val){
        T[] newArray = Arrays.copyOf(arr, arr.length+1);
        newArray[newArray.length - 1] = val;

        return newArray;
    }

    public static int findIndex(double[] arr, double val){
        for(int i = 0; i < arr.length; i++) if(arr[i] == val) return i;
        return -1;
    }

    public static int findIndexOfMax(double[] arr){
        double max = arr[0];
        int maxi = 0;

        for(int i = 1; i < arr.length; i++) if(arr[i] > max) {max = arr[i]; maxi = i;}

        return maxi;
    }

    public static int product(int[] arr){
        return Arrays.stream(arr).reduce(1, (i, j) -> i * j);
    }

    public static double sum(double[] arr){
        return Arrays.stream(arr).reduce(0, Double::sum);
    }

    public static int[] randomOrderInts(int min, int max){
        List<Integer> ints = IntStream.range(min, max + 1).boxed().collect(Collectors.toList());
        Collections.shuffle(ints);

        return ints.stream().mapToInt(i -> i).toArray();
    }

}
