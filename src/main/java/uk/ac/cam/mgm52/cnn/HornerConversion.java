package uk.ac.cam.mgm52.cnn;

/**Convert between coordinates and indices Horner's scheme*/
public class HornerConversion {

    //Declaring constructor as private to make the class effectively static
    private HornerConversion(){}

    /**Convert from coords to an index, using formula i = i1 + d1*(i2 + d2*(i3 + ...*/
    public static int coordsToHorner(int[] coords, int[] myDimSizes){
        int horner = 0;
        int product = 1;
        for(int i = 0; i < coords.length; i++){
            horner += coords[i] * product;
            product *= myDimSizes[i];
        }
        return horner;
    }

    /**Convert from index to coords, working back from formula i = i1 + d1*(i2 + d2*(i3 + ...*/
    public static int[] hornerToCoords(int hornerIndex, int[] myDimSizes){
        int[] coords = new int[myDimSizes.length];

        for(int i = 0; i < coords.length; i++){
            coords[i] = hornerIndex % myDimSizes[i];
            hornerIndex = (hornerIndex - coords[i])/myDimSizes[i];
        }

        return coords;
    }

}
