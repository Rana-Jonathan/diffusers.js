import * as ort from 'onnxruntime-web/webgpu';
// import * as ORT from '@aislamov/onnxruntime-web64'
// import type { InferenceSession } from 'onnxruntime-common'
import { replaceTensors } from '@/util/Tensor'
import { Tensor } from '@xenova/transformers'

ort.env.wasm.numThreads = 1;
ort.env.wasm.simd = true;

const isNode = typeof process !== 'undefined' && process?.release?.name === 'node'

let onnxSessionOptions = isNode
  ? {
    executionProviders: ['cpu'],
    executionMode: 'parallel',
  }
  : {
    executionProviders: ['webgpu'],
    enableMemPattern: false,
    enableCpuMemArena: false,
    extra: {
        session: {
            disable_prepacking: "1",
            use_device_allocator_for_initializers: "1",
            use_ort_model_bytes_directly: "1",
            use_ort_model_bytes_for_initializers: "1"
        }
    },
  }
 
export class Session {
  private session: ort.InferenceSession
  public config: Record<string, unknown>

  constructor (session: ort.InferenceSession, config: Record<string, unknown> = {}) {
    this.session = session
    this.config = config || {}
  }

  static async create (
    modelBuffer: ArrayBuffer,
    weightsBuffer?: ArrayBuffer,
    weightsFilename?: string,
    config?: Record<string, unknown>,
    options?: ort.InferenceSession.SessionOptions,
  ) {
    // const arg = typeof modelOrPath === 'string' ? modelOrPath : new Uint8Array(modelOrPath)
    // const arg = new Uint8Array(modelBuffer)

    const sessionOptions = {
      ...onnxSessionOptions,
      ...options,
    }

    // const executionProviders = sessionOptions.executionProviders.map((provider) => {
    //   if (typeof provider === 'string') {
    //     return {
    //       name: provider,
    //       ...weightsParams,
    //     }
    //   }

    //   return {
    //     ...provider,
    //     ...weightsParams,
    //   }
    // })

    const modelOptions = {
      externalData: [
        {
          data: weightsBuffer,
          path: weightsFilename,
        }
      ],
      // freeDimensionOverrides: {}
    }

    // @ts-ignore
    const session = await ort.InferenceSession.create(modelBuffer, {
      ...sessionOptions,
      ...modelOptions,
    })

    return new Session(session, config)
  }

  async run (inputs: Record<string, Tensor>) {
    // @ts-ignore
    const result = await this.session.run(inputs)
    return replaceTensors(result)
  }

  release () {
    return this.session.release()
  }
}
