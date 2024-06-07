// @ts-nocheck
import { PNDMScheduler, PNDMSchedulerConfig } from '@/schedulers/PNDMScheduler'
import { CLIPTokenizer } from '../tokenizers/CLIPTokenizer'
import { randomNormalTensor } from '@/util/Tensor'
import { Tensor, cat, mean } from '@xenova/transformers'
import { dispatchProgress, loadModel, PretrainedOptions, ProgressCallback, ProgressStatus, sessionRun } from './common'
import { getModelFile, getModelJSON } from '../hub'
import { Session } from '../backends'
import { GetModelFileOptions } from '@/hub/common'
import { PipelineBase } from '@/pipelines/PipelineBase'
import { LCMScheduler } from '@/schedulers/LCMScheduler'

export interface StableDiffusionXLInput {
  prompt: string
  negativePrompt?: string
  guidanceScale?: number
  seed?: string
  width?: number
  height?: number
  numInferenceSteps: number
  hasTimestepCond?: boolean
  sdV1?: boolean
  progressCallback?: ProgressCallback
  runVaeOnEachStep?: boolean
  img2imgFlag?: boolean
  inputImage?: Float32Array
  strength?: number
}

export class StableDiffusionXLPipeline extends PipelineBase {
  public textEncoder2: Session
  public tokenizer2: CLIPTokenizer
  declare scheduler: PNDMScheduler

  constructor (
    unet: Session,
    vaeEncoder: Session,
    vaeDecoder: Session,
    textEncoder: Session,
    textEncoder2: Session,
    tokenizer: CLIPTokenizer,
    tokenizer2: CLIPTokenizer,
    scheduler: PNDMScheduler,
  ) {
    super()
    this.unet = unet
    this.vaeEncoder = vaeEncoder
    this.vaeDecoder = vaeDecoder
    this.textEncoder = textEncoder
    this.textEncoder2 = textEncoder2
    this.tokenizer = tokenizer
    this.tokenizer2 = tokenizer2
    this.scheduler = scheduler
    this.vaeScaleFactor = 8
  }

  static createScheduler (config: PNDMSchedulerConfig) {
    return new PNDMScheduler(
      {
        prediction_type: 'epsilon',
        ...config,
      },
    )
  }

  static async fromPretrained (modelRepoOrPath: string, options?: PretrainedOptions) {
    const opts: GetModelFileOptions = {
      ...options,
    }

    const tokenizer = await CLIPTokenizer.from_pretrained(modelRepoOrPath, { ...opts, subdir: 'tokenizer' })
    const tokenizer2 = await CLIPTokenizer.from_pretrained(modelRepoOrPath, { ...opts, subdir: 'tokenizer_2' })

    const unet = await loadModel(
      modelRepoOrPath,
      'unet/model.onnx',
      opts,
    )
    const textEncoder2 = await loadModel(modelRepoOrPath, 'text_encoder_2/model.onnx', opts)
    const textEncoder = await loadModel(modelRepoOrPath, 'text_encoder/model.onnx', opts)
    const vaeEncoder = await loadModel(modelRepoOrPath, 'vae_encoder/model.onnx', opts)
    const vae = await loadModel(modelRepoOrPath, 'vae_decoder/model.onnx', opts)

    const schedulerConfig = await getModelJSON(modelRepoOrPath, 'scheduler/scheduler_config.json', true, opts)
    const scheduler = StableDiffusionXLPipeline.createScheduler(schedulerConfig)

    await dispatchProgress(opts.progressCallback, {
      status: ProgressStatus.Ready,
    })
    return new StableDiffusionXLPipeline(unet, vaeEncoder, vae, textEncoder, textEncoder2, tokenizer, tokenizer2, scheduler)
  }

  /**
   * Tokenizes and encodes the input prompt. Before encoding, we verify if it is necessary
   * to break the prompt into chunks due to the Tokenizer model limit (which is usually 77)
   * by getting the maximum length of the input prompt and comparing it with the Tokenizer
   * model max length. If the prompt exceeds the Tokenizer model limit, then it is
   * necessary to break the prompt into chunks, otherwise, it is not necessary.
   * 
   * @param prompt Input prompt.
   * @param tokenizer Tokenizer model.
   * @param textEncoder Text Encoder model.
   * @param highestTokenLength  Highest token length between prompt and negative prompt or Tokenizer model max length.
   * @param is64 Is 64 bit flag.
   * @returns Tensor containing the prompt embeddings.
   */
  async encodePromptXl (prompt: string, tokenizer: CLIPTokenizer, textEncoder: Session, highestTokenLength: number, is64: boolean = false) {
    let tokens, encoded, inputIds;
    const TokenMaxLength = tokenizer.model_max_length // Tokenizer model max length of tokens including the <START> and <END> tokens

    if(highestTokenLength > TokenMaxLength) {
      let embeddingsTensorArray: Tensor[] = [] // Will contain all of the prompt token embedding chunks
      const userTokenMaxLength = TokenMaxLength - 2 // Max length of tokens minus the <START> and <END> tokens

      tokens = this.tokenizer(
        prompt,
        {
          return_tensor: false,
          padding: false,
          max_length: TokenMaxLength,
          return_tensor_dtype: 'int32',
        },
      )

      inputIds = tokens.input_ids // Tokenized prompt
      const START_token = inputIds.shift() // Remove <START> token
      const END_token = inputIds.pop() // Remove <END> token

      for(let i = 0; i < highestTokenLength; i += userTokenMaxLength) {
        let tokenChunk = inputIds.slice(i, i + userTokenMaxLength)

        for(let j = tokenChunk.length; j < userTokenMaxLength; j++) { // Pad chunk to userTokenMaxLength if necessary. Use the <END> token to pad.
          tokenChunk.push(END_token)
        }

        tokenChunk.unshift(START_token) // Add <START> token to each chunk
        tokenChunk.push(END_token) // Add <END> token to each chunk

        const tensor = 
        is64 
        ? new Tensor('int64', BigInt64Array.from(tokenChunk.flat().map(x => BigInt(x))), [1, tokenChunk.length])
        : new Tensor('int32', Int32Array.from(tokenChunk.flat()), [1, tokenChunk.length])
  
        // @ts-ignore
        encoded = await sessionRun(textEncoder, { input_ids: tensor })
        delete encoded.pooler_output // deleted as it is not used and creates conflict when concatenating later on
        embeddingsTensorArray.push(encoded)
      }

      const embeddingsTensorArrayKeys = Object.keys(embeddingsTensorArray[0]) // get keys of elements to concatenate one by one
      let objectTensorToReturn = {} // object containing the concatenated embeddings
      
      // we have to concatenate each hidden_state element of each chunk one by one to achieve complete concatenation
      embeddingsTensorArrayKeys.forEach(key => { // for every key in every element in embeddingsTensorArray
        let arrayToConcat = [] // temp array used for concatenation

        embeddingsTensorArray.forEach(element => { // for every element in embeddingsTensorArray
          arrayToConcat.push(element[key])
        });

        /** 
         * the text_embeds element is different in shape than the hidden_state elements. This difference creates an issue when
         * concatenating the 1 dimension and I was not able to find documentation about concatenating text embeddings. What was
         * most logical to me was to concatenate these embeddings and getting the mean along the 0 dimension in order to get
         * the expected output shape [1, 1280]. Tested this and it worked.
        */
        objectTensorToReturn[key] = key == 'text_embeds' ? mean(cat(arrayToConcat), 0, true) : cat(arrayToConcat, 1);
      });

      return objectTensorToReturn
    }
    else {
      const tokens = tokenizer(
        prompt,
        {
          return_tensor: false,
          padding: true,
          max_length: tokenizer.model_max_length,
          return_tensor_dtype: 'int32',
        },
      )
  
      const inputIds = tokens.input_ids
      const tensor = 
        is64 
        ? new Tensor('int64', BigInt64Array.from(inputIds.flat().map(x => BigInt(x))), [1, inputIds.length])
        : new Tensor('int32', Int32Array.from(inputIds.flat()), [1, inputIds.length])
  
      // @ts-ignore
      return await sessionRun(textEncoder, { input_ids: tensor })
    }
  }

  /**
   * Returns the prompt and negative prompt text embeddings.
   * 
   * @param prompt Input prompt.
   * @param negativePrompt Input negative prompt.
   * @returns Tensor containing the prompt and negative prompt embeddings.
   */
  async getPromptEmbedsXl (prompt: string, negativePrompt: string|undefined) {
    // We check which has more tokens between the prompt and negative prompt
    const promptTokens = this.tokenizer(
      prompt,
      {
        return_tensor: false,
        padding: false,
        max_length: this.tokenizer.model_max_length,
        return_tensor_dtype: 'int32',
      },
    )

    const negPromptTokens = this.tokenizer(
      negativePrompt,
      {
        return_tensor: false,
        padding: false,
        max_length: this.tokenizer.model_max_length,
        return_tensor_dtype: 'int32',
      },
    )

    const promptTokensLength = promptTokens.input_ids.length // Number of tokens in prompt including the <START> and <END> tokens
    const negPromptTokensLength = negPromptTokens.input_ids.length // Number of tokens in negative prompt including the <START> and <END> tokens
    const highestTokenLength = Math.max(promptTokensLength, negPromptTokensLength)

    const promptEmbeds = await this.encodePromptXl(prompt, this.tokenizer, this.textEncoder, highestTokenLength, false)
    let num1HiddenStates = 0

    for (let i = 0; i < 100; i++) {
      if (promptEmbeds[`hidden_states.${i}`] === undefined) {
        break
      }

      num1HiddenStates++
    }

    let posHiddenStates = promptEmbeds[`hidden_states.${num1HiddenStates - 2}`]

    let negHiddenStates

    if (negativePrompt) {
      const negativePromptEmbeds = await this.encodePromptXl(negativePrompt || '', this.tokenizer, this.textEncoder, highestTokenLength)
      negHiddenStates = negativePromptEmbeds[`hidden_states.${num1HiddenStates - 2}`]
    }

    const promptEmbeds2 = await this.encodePromptXl(prompt, this.tokenizer2, this.textEncoder2, highestTokenLength, true)

    let num2HiddenStates = 0
    for (let i = 0; i < 100; i++) {
      if (promptEmbeds2[`hidden_states.${i}`] === undefined) {
        break
      }

      num2HiddenStates++
    }

    posHiddenStates = cat([posHiddenStates, promptEmbeds2[`hidden_states.${num2HiddenStates - 2}`]], -1)
    const posTextEmbeds = promptEmbeds2.text_embeds
    let negTextEmbeds

    if (negativePrompt) {
      const negativePromptEmbeds2 = await this.encodePromptXl(negativePrompt || '', this.tokenizer2, this.textEncoder2, highestTokenLength, true)
      negHiddenStates = cat([negHiddenStates, negativePromptEmbeds2[`hidden_states.${num2HiddenStates - 2}`]], -1)
      negTextEmbeds = negativePromptEmbeds2.text_embeds
    } else {
      negHiddenStates = posHiddenStates.mul(0)
      negTextEmbeds = posTextEmbeds.mul(0)
    }

    return {
      positive: {
        lastHiddenState: posHiddenStates,
        textEmbeds: posTextEmbeds,
      },
      negative: {
        lastHiddenState: negHiddenStates,
        textEmbeds: negTextEmbeds,
      },
    }
  }

  getTimeEmbeds (width: number, height: number) {
    return new Tensor(
      'float32',
      [height, width, 0, 0, height, width],
      [1, 6],
    )
  }

  async run (input: StableDiffusionXLInput) {
    const width = input.width || 1024
    const height = input.height || 1024
    const batchSize = 1
    const guidanceScale = input.guidanceScale || 5
    const seed = input.seed || ''

    this.scheduler.setTimesteps(input.numInferenceSteps || 5)

    await dispatchProgress(input.progressCallback, {
      status: ProgressStatus.EncodingPrompt,
    })

    const hasGuidance = guidanceScale >= 0.01
    const promptEmbeds = await this.getPromptEmbedsXl(input.prompt, hasGuidance ? input.negativePrompt : '')
    
    const latentShape = [batchSize, 4, width / 8, height / 8]
    let latents = randomNormalTensor(latentShape, undefined, undefined, 'float32', seed) // Normal latents used in Text-to-Image
    let denoised: Tensor
    const timesteps = this.scheduler.timesteps.data
    let humanStep = 1
    let cachedImages: Tensor[]|null = null

    const timeIds = this.getTimeEmbeds(width, height)

    for (const step of timesteps) {
      const timestep = new Tensor(new BigInt64Array([BigInt(step)]))
      await dispatchProgress(input.progressCallback, {
        status: ProgressStatus.RunningUnet,
        unetTimestep: humanStep,
        unetTotalSteps: timesteps.length,
      })
      
      const textNoise = await this.unet.run(input.hasTimestepCond
        ? {
          sample: latents,
          timestep,
          timestep_cond: randomNormalTensor([1, 256], undefined, undefined, 'float32', seed),
          encoder_hidden_states: promptEmbeds.positive.lastHiddenState,
          text_embeds: promptEmbeds.positive.textEmbeds,
          time_ids: timeIds,
        }
        : {
          sample: latents,
          timestep,
          encoder_hidden_states: promptEmbeds.positive.lastHiddenState,
          text_embeds: promptEmbeds.positive.textEmbeds,
          time_ids: timeIds,
        },
      )

      let noisePred

      if (hasGuidance) {
        const uncondNoise = await this.unet.run(input.hasTimestepCond
          ? {
            sample: latents,
            timestep,
            encoder_hidden_states: promptEmbeds.negative.lastHiddenState,
            timestep_cond: randomNormalTensor([1, 256], undefined, undefined, 'float32', seed),
            text_embeds: promptEmbeds.negative.textEmbeds,
            time_ids: timeIds,
          }
          : {
            sample: latents,
            timestep,
            encoder_hidden_states: promptEmbeds.negative.lastHiddenState,
            text_embeds: promptEmbeds.negative.textEmbeds,
            time_ids: timeIds,
          })

        const noisePredUncond = uncondNoise.out_sample
        const noisePredText = textNoise.out_sample
        noisePred = noisePredUncond.add(noisePredText.sub(noisePredUncond).mul(guidanceScale))
      }
      else {
        noisePred = textNoise.out_sample
      }

      const schedulerOutput = this.scheduler.step(
        noisePred,
        step,
        latents,
      )

      latents = schedulerOutput
      denoised = schedulerOutput

      if (this.scheduler instanceof LCMScheduler) {
        latents = schedulerOutput[0]
        denoised = schedulerOutput[1]
      }

      if (input.runVaeOnEachStep) {
        await dispatchProgress(input.progressCallback, {
          status: ProgressStatus.RunningVae,
          unetTimestep: humanStep,
          unetTotalSteps: timesteps.length,
        })
        cachedImages = await this.makeImages(denoised)
      }
      humanStep++
    }

    if (input.runVaeOnEachStep) {
      return cachedImages!
    }

    return this.makeImages(denoised)
  }

  async release () {
    await super.release()
    return this.textEncoder2?.release()
  }
}
