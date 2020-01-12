package uk.ac.cam.mgm52.cnn;

import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

public final class ArrayUtils {

    private ArrayUtils(){}

    //Elementwise addition
    public static int[] addAll(int[] arr, int[] arr2){
        int maxLength = Math.max(arr.length, arr2.length);

        int[] result = new int[maxLength];
        for(int i = 0; i < maxLength; i++){
            result[i] = (arr.length>i?arr[i]:0) + (arr2.length>i?arr2[i]:0);
        }
        return result;
    }

    public static double[] addAll(double[] arr, double[] arr2){
        int maxLength = Math.max(arr.length, arr2.length);

        double[] result = new double[maxLength];
        for(int i = 0; i < maxLength; i++){
            result[i] = (arr.length>i?arr[i]:0) + (arr2.length>i?arr2[i]:0);
        }
        return result;
    }

    //Add val to all elements of arr
    public static int[] addAll(int[] arr, int val){
        int[] result = new int[arr.length];
        for(int i = 0; i < arr.length; i++){
            result[i] = arr[i] + val;
        }
        return result;
    }

    public static int[] subtractAll(int[] arr, int[] arr2){
        int maxLength = Math.max(arr.length, arr2.length);

        int[] result = new int[maxLength];
        for(int i = 0; i < maxLength; i++){
            result[i] = (arr.length>i?arr[i]:0) - (arr2.length>i?arr2[i]:0);
        }
        return result;
    }

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

        for(int i = 1; i < arr.length; i++) if(arr[i] > max
        ) {max = arr[i]; maxi = i;}

        return maxi;
    }

    public static int[] divideAll(int[] arr, int[] arr2){
        int[] result = new int[arr.length];

        for(int i = 0; i < arr.length; i++){
            result[i] = arr[i] / arr2[i];
        }
        return result;
    }

    public static double[] divideAll(double[] arr, double val){
        double[] result = new double[arr.length];

        for(int i = 0; i < arr.length; i++){
            result[i] = arr[i] / val;
        }
        return result;
    }

    public static int[] multAll(int[] arr, int[] arr2){
        int[] result = new int[arr.length];

        for(int i = 0; i < arr.length; i++){
            result[i] = arr[i] * arr2[i];
        }
        return result;
    }

    public static double[] multAll(double[] arr, double[] arr2){
        double[] result = new double[arr.length];

        for(int i = 0; i < arr.length; i++){
            result[i] = arr[i] * arr2[i];
        }
        return result;
    }

    public static double[] multAll(double[] arr, double val){
        double[] result = new double[arr.length];

        for(int i = 0; i < arr.length; i++){
            result[i] = arr[i] * val;
        }
        return result;
    }

    public static int product(int[] arr){
        return Arrays.stream(arr).reduce(1, (i, j) -> i * j);
    }

    public static double sum(double[] arr){
        return Arrays.stream(arr).reduce(0, (i, j) -> i + j);
    }

    public static int[] randomOrderInts(int min, int max){
        List<Integer> ints = IntStream.range(min, max + 1).boxed().collect(Collectors.toList());
        Collections.shuffle(ints);

        return ints.stream().mapToInt(i -> i).toArray();
    }

}
