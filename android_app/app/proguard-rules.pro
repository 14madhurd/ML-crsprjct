# OSMDroid ProGuard rules
-keep interface org.osmdroid.** { *; }
-keep class org.osmdroid.** { *; }
-dontwarn org.osmdroid.**

# Google Play Services Location
-keep class com.google.android.gms.location.** { *; }
