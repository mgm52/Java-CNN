package uk.ac.cam.mgm52.cnn;

public final class ArrayUtils {

    private ArrayUtils(){}

    //Elementwise addition
    public static int[] addAll(int[] arr, int[] arr2){
        int[] result = new int[arr.length];
        for(int i = 0; i < arr.length; i++){
            result[i] = arr[i] + arr2[i];
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
        int[] result = new int[arr.length];
        for(int i = 0; i < arr.length; i++){
            result[i] = arr[i] - arr2[i];
        }
        return result;
    }

}
