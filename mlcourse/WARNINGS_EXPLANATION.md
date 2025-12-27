# Warnings in TF-IDF.py - Explanation and Resolution

## Warnings That Cannot Be Fixed (System/Java Level)

These warnings come from the Java Virtual Machine (JVM) and PySpark compatibility issues with newer Java versions. They are **harmless** and don't affect functionality:

1. **`WARNING: Using incubator modules: jdk.incubator.vector`**
   - **Cause**: Java 17+ uses incubator modules
   - **Impact**: None - just informational
   - **Fix**: Cannot be fixed - this is a Java version compatibility notice

2. **`WARNING: package sun.security.action not in java.base`**
   - **Cause**: Java module system changes in newer versions
   - **Impact**: None - Spark handles this internally
   - **Fix**: Cannot be fixed - Java version compatibility

3. **`WARNING: A terminally deprecated method in sun.misc.Unsafe has been called`**
   - **Cause**: PySpark uses deprecated Java internal APIs that will be removed in future Java versions
   - **Impact**: None currently, but may break in future Java versions
   - **Fix**: Cannot be fixed - requires PySpark update to use newer APIs

4. **`WARNING: sun.misc.Unsafe::arrayBaseOffset will be removed in a future release`**
   - **Cause**: Same as above - deprecated Java API
   - **Impact**: None currently
   - **Fix**: Cannot be fixed - requires PySpark update

## Warnings That Can Be Partially Suppressed

5. **`WARN NativeCodeLoader: Unable to load native-hadoop library`**
   - **Cause**: Hadoop native libraries not available for macOS
   - **Impact**: None - Spark falls back to Java implementations (slightly slower but works fine)
   - **Fix**: Can be suppressed by setting log level, but warning may still appear during initialization

6. **`WARN Utils: Your hostname resolves to a loopback address`**
   - **Cause**: Hostname resolution issue
   - **Impact**: None - Spark automatically uses the correct IP
   - **Fix**: Can be suppressed by setting `SPARK_LOCAL_IP` environment variable (already done in code)

## Summary

**All warnings are harmless** and don't affect the script's functionality. The script works correctly despite these warnings.

**To reduce visible warnings:**
- The script already sets log level to ERROR for Spark operations
- The script already sets SPARK_LOCAL_IP to suppress hostname warnings
- Java-level warnings cannot be suppressed as they come from the JVM itself

**Recommendation**: These warnings are safe to ignore. They're informational messages about Java/PySpark compatibility with newer Java versions.

