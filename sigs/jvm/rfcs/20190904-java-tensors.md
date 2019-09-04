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
in order to allow user to read or write tensor data from the Java space, the `Tensor` class must implement the 
new `NdArray` interface. This interface carries as a generic parameter the Java type used to access this data. 
For example, a `NdArray<Integer>` can read or write Java `Integer`s in a n-dimensional data structure.

Actually the `Tensor` class also have a generic parameter, that looks similar to the one used by the `NdArray` but 
for a completely different purpose. This parameter is only present to check compile-time type compatibility between 
operands of an operation. For example, a `Tensor<Integer>` can be added to another `Tensor<Integer>` (via `tf.constant` 
and `tf.math.add`) but not to a `Tensor<Float>`.

We can naively think that the Tensor type parameter can be used to serve both purpose (i.e. compile-time type safety
and data I/O) but it is not possible because there is no guarantee that its value matches a type that can be used
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


## Detailed Design

## Questions and Discussion Topics
