# Java Tensors
| Status        | Proposed       |
:-------------- |:---------------------------------------------------- |
| **Author(s)** | Karl Lessard (karl.lessard@gmail.com) |
| **Updated**   | 2019-09-04                                           |

## Objective

Expose differently tensors from the TensorFlow Java client to support more data types while allowing the user
to manipulate their content directly from Java space.

## Motivation

In order to implement [previous RFC](https://github.com/karllessard/community/blob/master/sigs/jvm/rfcs/20190606-java-tensor-io.md)
about I/O operations on tensors from Java space, some changes must be made the the actual Tensor API found in 
the TensorFlow Java core client.

But there are other reasons as well to justify a full redesign of that API. By allowing the first 2.x TF Java
release to break backward compatibility with previous versions, we could take that opportunity to fix most of
the limitations of the current API.

## User Benefit

The new API will be prepared to support more data types, while still allowing the user to read and write to tensors
directly from the Java space and enforcing compile-time type safety.

## Design Proposal

### Background

As planned in the [Java Tensor NIO RFC](https://github.com/karllessard/community/blob/master/sigs/jvm/rfcs/20190606-java-tensor-io.md), 
 to allow user to read or write tensor data from the Java space, the `Tensor` class must implement the 
new `NdArray` interface. This interface carries as a generic parameter the Java type used to access this data. 
For example, a `NdArray<Integer>` can read or write Java `Integer`s in a n-dimensional data structure.

Actually the `Tensor` class also have a generic parameter, that looks similar to the one used by the `NdArray` but 
for a completely different purpose. This parameter is only present to check compile-time type compatibility between 
operands of an operation. For example, a `Tensor<Integer>` can be added to another `Tensor<Integer>` (via `tf.constant` 
and `tf.math.add`) but not to a `Tensor<Float>`.

We can naively think that the Tensor type parameter can be used to serve both purposes (i.e. compile-time type safety
and direct data access) but it is not possible because there is no guarantee that its value matches a type that can be used
to store data in memory. In fact, the `Integer` value in the previous example is completely unrelated with what 
an integer is in Java, it is simply used as an idiomatic alias to `INT32` TF data type. For this reason, you can
have custom type classes, such as `Tensor<UInt8>`, that insure type safety but you cannot read a `UInt8`
value from memory, you actually read a `Byte`. 

Since the parameter of the `NdArray` must be the Java type of the data stored in memory, since `Tensor` needs
to implement the `NdArray` interface to allow direct I/O operations, and since we carry the TF data type as a
parameter to `Tensor` for type checking, then we would end up needing two generic types in the `Tensor`
signature, like `Tensor<UInt8, Byte>` or `Tensor<Integer, Integer>`. 

This can start to be annoying pretty fast. The current proposal main purpose is to avoid the hassle of carrying
to much information in generic parameters by taking a completely different approach.

### Tensor Types

#### Tensor Classes

A solution is to identify tensor data types not by a type parameter but by a concrete class. There would be
one class per supported data type, which extends from the base class `Tensor<T>`, where `T` is the Java type
used for data access (therefore is used to implement `NdArray<T>`).

In addition, having our own type concrete classes allows us to group those types into families that can be
used to add more contraints on which type of tensor is allowed for a given TF operation (e.g. only `Numeric`
data types can be used in a `tf.math.add` operation).

Let see what it looks like in a quick example:

```java
public abstract class Tensor<T> implements NdArray<T> { ... }

public interface Numeric {}

public final class TInt32 extends Tensor<Integer> implements Numeric { ... }

public final class TUInt8 extends Tensor<Byte> implements Numeric { ... }

public final class TString extends Tensor<String> { ... }
```

To support compile-time type safety in the operands of a TF operations, we can carry that concrete type name
as a parameter to the operations (instead of the actual idiomatic aliases).

```java
Constant<TInt32> c = tf.constant(2); // more on this constant creation in the next section...

Add<TInt32> addOp = tf.math.add(c, c);

TInt32 addResult = addOp.tensor(); // only possible in eager mode

Integer sum = addResult.get(); // from NdArray<Integer> interface
```

The last lines are interesting, as in eager mode, we can easily convert the result of an operation to a tensor
of the appropriate type and then read its data. We can also achieve something similar when fetching results
of a graph session.

#### Data Types

Now even with those new concrete classes, we need to be able to carry information about a given data type without
the need of instantiating a tensor. Actually this information is carried by the [`DataType` enum class](https://github.com/karllessard/tensorflow/blob/master/tensorflow/java/src/main/java/org/tensorflow/DataType.java),
which converts back and forth a type alias (such as `Integer`) to a TF data type (such as `INT32`).

Again, this conversion can be sometimes painful and could be avoid by changing the `DataType` enum class to a 
normal one, that is instantiated in each tensor class. For example:

```java
public abstract class DataType<U> {

    int ordinal();  // this is the numeric value used for this datatype by the C++ core
    
    int byteSize();  // number of bytes required per value, -1 meaning undefined
}

public class TInt32 extends Tensor<Integer> implements Numeric {

    public static final DataType<TInt32> DTYPE = DataType.create(TInt32.class, 3, 4);
}
```

This way, you can pass a data type as an attribute to an operation by accessing directly the `DTYPE` member
of the tensor class that represents this type, with no enum conversion needed. For example, 
`Placeholder<TInt32> p = tf.placeholder(TInt32.DTYPE)`.

### Tensor Allocation

There is actually a plenitude of methods to allocate a tensor in Java.

First, there is a bunch of [factories](https://github.com/karllessard/tensorflow/blob/44ebaf12e7f196ef621413ce90c48bb9ae5f7522/tensorflow/java/src/main/java/org/tensorflow/Tensor.java#L105)
found in the [`Tensor` class](https://github.com/karllessard/tensorflow/blob/master/tensorflow/java/src/main/java/org/tensorflow/Tensor.java)
directly that accept any object as data input. As discussed in [Java Tensor NIO RFC](https://github.com/karllessard/community/blob/master/sigs/jvm/rfcs/20190606-java-tensor-io.md), while very
flexible, those factories make use of heavy reflection techniques that offer poor performances and must be avoided.
[Other factories](https://github.com/karllessard/tensorflow/blob/44ebaf12e7f196ef621413ce90c48bb9ae5f7522/tensorflow/java/src/main/java/org/tensorflow/Tensor.java#L167)
accept a `java.nio.Buffer` for data input, which proved to be way more efficient but is not
easy to use, especially when dealing with high rank tensors. Those factories were already planned to be replaced
by an equivalent that accept a `NdArray` instead.

Then there is a multitude of factories found in the [`Tensors` helper class](https://github.com/karllessard/tensorflow/blob/master/tensorflow/java/src/main/java/org/tensorflow/Tensors.java),
which make all use of the non-efficient reflective allocators discussed before, so they must be avoided in their
actual form as well.

Finally, the [`Constant` operation wrapper](https://github.com/karllessard/tensorflow/blob/master/tensorflow/java/src/main/java/org/tensorflow/op/core/Constant.java)
offers also a convenient way to quickly allocate a single-use tensor for creating a constant, most are still based on 
the same non-efficient reflective allocators.

For a better experience, it should be more intuitive for a user to know which factory to use and when. Basically, there
is two main cases to cover: allocating a tensor for creating a constant and allocating a tensor for feeding data
to the network.

#### Data Tensors

Data tensors, such as those feeding a network, must be allocated explicitely by the user by 

#### Constant Tensors

For constant allocation, the actual form of accepting a Java constant in parameter to the `tf.constant()` method is 
very intuitive and concise. For example, `tf.constant(0)`, where `tf` is an instance of `Ops`.
We should not introduce more complexity unless needed. Plus, it can be generally safe
to assume that when dealing with constants, we can continue to map implicitely a given Java type to a TF data type
(e.g. a `int` value generates a `TInt32` constant, a `float` value generates a `TFloat` constant, etc.)

The underlying implementation though must be changed to avoid using reflective techniques to generate the tensor and
must be more explicit. To simplify this task, we should limit the possible "short-cut" factories to rank-0 or rank-1
constants (which satisfies most of the cases) and rely on explicit `Tensor` for constant of a higher rank. For example:

```java
public final class Constant<U> extends PrimitiveOp {

    public static final Constant<TInt32> create(Scope scope, Integer value) {
        ...
    }
    
    public static final Constant<TInt32> create(Scope scope, Integer... values) {
        ...
    }
    
    public static final <U extends Tensor<?>> Constant<U> create(Scope scope, U data) {
        ...
    }
}

// Usage with the Ops API

Constant<TInt32> scalar = tf.constant(10);  // single argument constructor is chosen

Constant<TInt32> vector = tf.constant(10, 11, 42);  // variadic constructor is chosen

Constant<TInt32> matrix = tf.constant(TInt32.ofShape(2, 2).row(10, 11).row(30, 40).done());

```




## Detailed Design

## Questions and Discussion Topics
