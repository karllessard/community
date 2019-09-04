# Java Tensors
| Status        | Proposed       |
:-------------- |:---------------------------------------------------- |
| **Author(s)** | Karl Lessard (karl.lessard@gmail.com) |
| **Updated**   | 2019-09-04                                           |

## Objective

Expose differently tensors from the TensorFlow Java client to support more data types while allowing the user
to manipulate their content directly from Java space.

## Motivation

In order to implement [previous RFC](https://github.com/karllessard/community/blob/master/sigs/jvm/rfcs/20190606-java-tensor-io.md) about I/O operations on tensors from Java space, some changes must be made the the actual Tensor API found in the TensorFlow Java core client.

But there are other reasons as well to justify a full redesign of that API. By allowing the first 2.x TF Java release to break backward compatibility with previous versions, we could take that opportunity to fix most of the limitations of the current API.

## User Benefit

The new API will be prepared to support more data types, while still allowing the user to read and write to tensors directly from the Java space and enforcing compile-time type safety.

## Design Proposal

### Tensor API

## Detailed Design

## Questions and Discussion Topics
