package com.glec.dtg.common

/**
 * GLEC DTG - Type-Safe Result Container
 *
 * Production-grade error handling using sealed classes for type safety.
 * Eliminates null checks and exception handling boilerplate.
 *
 * Usage:
 * ```kotlin
 * fun readSensor(): Result<Float, SensorError> {
 *     return try {
 *         val value = sensor.read()
 *         Result.Success(value)
 *     } catch (e: Exception) {
 *         Result.Failure(SensorError.ReadFailed(e))
 *     }
 * }
 *
 * // Pattern matching with when
 * when (val result = readSensor()) {
 *     is Result.Success -> Log.d(TAG, "Value: ${result.value}")
 *     is Result.Failure -> Log.e(TAG, "Error: ${result.error}")
 * }
 *
 * // Functional transformations
 * readSensor()
 *     .map { it * 2 }
 *     .flatMap { processValue(it) }
 *     .fold(
 *         onSuccess = { value -> displayValue(value) },
 *         onFailure = { error -> showError(error) }
 *     )
 * ```
 *
 * @param T Success value type
 * @param E Error type (should be sealed class)
 */
sealed class Result<T, E> {

    /**
     * Success case containing the result value
     */
    data class Success<T>(val value: T) : Result<T, Nothing>()

    /**
     * Failure case containing the error
     */
    data class Failure<E>(val error: E) : Result<Nothing, E>()

    // ========================================
    // Predicates
    // ========================================

    /**
     * Returns true if this is a Success
     */
    val isSuccess: Boolean
        get() = this is Success

    /**
     * Returns true if this is a Failure
     */
    val isFailure: Boolean
        get() = this is Failure

    // ========================================
    // Extractors (nullable)
    // ========================================

    /**
     * Returns the value if Success, null otherwise
     */
    fun getOrNull(): T? = when (this) {
        is Success -> value
        is Failure -> null
    }

    /**
     * Returns the error if Failure, null otherwise
     */
    fun errorOrNull(): E? = when (this) {
        is Success -> null
        is Failure -> error
    }

    // ========================================
    // Extractors (with default)
    // ========================================

    /**
     * Returns the value if Success, or the default value otherwise
     */
    fun getOrDefault(default: T): T = when (this) {
        is Success -> value
        is Failure -> default
    }

    /**
     * Returns the value if Success, or computes a default from the error
     */
    inline fun getOrElse(onFailure: (E) -> T): T = when (this) {
        is Success -> value
        is Failure -> onFailure(error)
    }

    // ========================================
    // Transformations
    // ========================================

    /**
     * Maps the success value using the given transformation
     */
    @Suppress("UNCHECKED_CAST")
    inline fun <R> map(transform: (T) -> R): Result<R, E> = when (this) {
        is Success -> Success(transform(value)) as Result<R, E>
        is Failure -> this as Result<R, E>
    }

    /**
     * Maps the error using the given transformation
     */
    @Suppress("UNCHECKED_CAST")
    inline fun <F> mapError(transform: (E) -> F): Result<T, F> = when (this) {
        is Success -> this as Result<T, F>
        is Failure -> Failure(transform(error)) as Result<T, F>
    }

    /**
     * FlatMap - chains another Result-producing operation
     */
    @Suppress("UNCHECKED_CAST")
    inline fun <R> flatMap(transform: (T) -> Result<R, E>): Result<R, E> = when (this) {
        is Success -> transform(value)
        is Failure -> this as Result<R, E>
    }

    /**
     * Recovers from failure by providing an alternative Result
     */
    inline fun recover(transform: (E) -> Result<T, E>): Result<T, E> = when (this) {
        is Success -> this
        is Failure -> transform(error)
    }

    // ========================================
    // Side Effects
    // ========================================

    /**
     * Executes the given action if this is a Success
     */
    inline fun onSuccess(action: (T) -> Unit): Result<T, E> {
        if (this is Success) {
            action(value)
        }
        return this
    }

    /**
     * Executes the given action if this is a Failure
     */
    inline fun onFailure(action: (E) -> Unit): Result<T, E> {
        if (this is Failure) {
            action(error)
        }
        return this
    }

    /**
     * Folds this Result into a single value using the provided functions
     */
    inline fun <R> fold(
        onSuccess: (T) -> R,
        onFailure: (E) -> R
    ): R = when (this) {
        is Success -> onSuccess(value)
        is Failure -> onFailure(error)
    }

    // ========================================
    // Companion Object - Builders
    // ========================================

    companion object {

        /**
         * Wraps a code block in a try-catch and returns a Result
         */
        inline fun <T, E> catch(
            errorMapper: (Throwable) -> E,
            block: () -> T
        ): Result<T, E> {
            return try {
                Success(block()) as Result<T, E>
            } catch (e: Throwable) {
                Failure(errorMapper(e)) as Result<T, E>
            }
        }

        /**
         * Wraps a code block that may return null into a Result
         */
        inline fun <T, E> fromNullable(
            value: T?,
            error: () -> E
        ): Result<T, E> {
            return if (value != null) {
                Success(value) as Result<T, E>
            } else {
                Failure(error()) as Result<T, E>
            }
        }

        /**
         * Converts a boolean to a Result
         */
        fun <E> fromBoolean(
            condition: Boolean,
            error: () -> E
        ): Result<Unit, E> {
            return if (condition) {
                Success(Unit) as Result<Unit, E>
            } else {
                Failure(error()) as Result<Unit, E>
            }
        }

        /**
         * Combines multiple Results into a single Result containing a list
         * Returns Failure if any Result is a Failure
         */
        @Suppress("UNCHECKED_CAST")
        fun <T, E> all(results: List<Result<T, E>>): Result<List<T>, E> {
            val values = mutableListOf<T>()
            for (result in results) {
                when (result) {
                    is Success -> values.add(result.value)
                    is Failure -> return result as Result<List<T>, E>
                }
            }
            return Success(values) as Result<List<T>, E>
        }
    }
}

// ========================================
// Extension Functions
// ========================================

/**
 * Converts a nullable value to a Result
 */
fun <T, E> T?.toResult(error: () -> E): Result<T, E> {
    return Result.fromNullable(this, error)
}

/**
 * Converts a Boolean to a Result<Unit, E>
 */
fun <E> Boolean.toResult(error: () -> E): Result<Unit, E> {
    return Result.fromBoolean(this, error)
}

/**
 * Flattens a nested Result
 */
fun <T, E> Result<Result<T, E>, E>.flatten(): Result<T, E> {
    return this.flatMap { it }
}
